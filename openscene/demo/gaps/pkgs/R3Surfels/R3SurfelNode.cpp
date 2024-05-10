/* Source file for the R3 surfel node class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// Namespace
////////////////////////////////////////////////////////////////////////

namespace gaps {



////////////////////////////////////////////////////////////////////////
// PRIVATE FLAGS
////////////////////////////////////////////////////////////////////////

#define R3_SURFEL_NODE_FLAGS_UPTODATE_FLAG             0x0004

#define R3_SURFEL_NODE_HAS_AERIAL_FLAG                 0x0010
#define R3_SURFEL_NODE_HAS_TERRESTRIAL_FLAG            0x0020
#define R3_SURFEL_NODE_HAS_ACTIVE_FLAG                 0x0040
#define R3_SURFEL_NODE_HAS_NORMALS_FLAG                0x0080



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelNode::
R3SurfelNode(const char *name)
  : object(NULL),
    scan(NULL),
    tree(NULL),
    tree_index(-1),
    parent(NULL),
    parts(),
    blocks(),
    complexity(-1),
    resolution(-1),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    elevation_range(FLT_MAX,-FLT_MAX),
    timestamp_range(FLT_MAX,-FLT_MAX),
    name((name) ? RNStrdup(name) : NULL),
    flags(0),
    data(NULL)
{
}



R3SurfelNode::
R3SurfelNode(const R3SurfelNode& node)
  : object(NULL),
    scan(NULL),
    tree(NULL),
    tree_index(-1),
    parent(NULL),
    parts(),
    blocks(node.blocks),
    complexity(node.complexity),
    resolution(node.resolution),
    bbox(node.bbox),
    timestamp_range(node.timestamp_range),
    name((node.name) ? RNStrdup(node.name) : NULL),
    flags(0),
    data(NULL)
{
}



R3SurfelNode::
~R3SurfelNode(void)
{
  // Remove blocks from node
  while (NBlocks() > 0) RemoveBlock(Block(NBlocks()-1));

  // Remove node from object
  if (object) object->RemoveNode(this);

  // Remove node from scan
  if (scan) scan->SetNode(NULL);

  // Delete parts
  while (NParts() > 0) delete (Part(NParts()-1));
  
  // Remove node from tree
  if (tree) tree->RemoveNode(this);

  // Delete name
  if (name) free(name);
}



////////////////////////////////////////////////////////////////////////
// PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////////////////

RNScalar R3SurfelNode::
Complexity(void) const
{
  // Return compexity (e.g., number of surfels in node)
  if (complexity < 0) ((R3SurfelNode *) this)->UpdateComplexity();
  return complexity;
}



RNScalar R3SurfelNode::
Resolution(void) const
{
  // Return resolution of node (surfels per area)
  if (resolution < 0) ((R3SurfelNode *) this)->UpdateResolution();
  return resolution;
}



RNLength R3SurfelNode::
AverageRadius(void) const
{
  // Sum average radius of surfels 
  RNLength total_radius = 0;
  RNLength total_weight = 0;
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    RNScalar weight = block->NSurfels();
    total_radius += weight * block->AverageRadius();
    total_weight += weight;
  }

  // Return weighted average
  return (total_weight > 0) ? total_radius / total_weight : 0;
}



const R3Box& R3SurfelNode::
BBox(void) const
{
  // Return bounding box of node
  if (bbox[0][0] == FLT_MAX) 
    ((R3SurfelNode *) this)->UpdateBBox();
  return bbox;
}



const RNInterval& R3SurfelNode::
ElevationRange(void) const
{
  // Return elevation interval of node
  if (elevation_range.Min() == FLT_MAX) 
    ((R3SurfelNode *) this)->UpdateElevationRange();
  return elevation_range;
}



const RNInterval& R3SurfelNode::
TimestampRange(void) const
{
  // Return timestamp interval of node
  if (timestamp_range.Min() == FLT_MAX) 
    ((R3SurfelNode *) this)->UpdateTimestampRange();
  return timestamp_range;
}

  

RNBoolean R3SurfelNode::
HasSurfels(RNBoolean leaf_level) const
{
  // Check blocks
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    if (block->NSurfels() > 0) return TRUE;
  }

  // Check parts
  if (leaf_level) {
    for (int i = 0; i < NParts(); i++) {
      R3SurfelNode *part = Part(i);
      if (part->HasSurfels(leaf_level)) return TRUE;
    }
  }

  // Did not find anything
  return FALSE;
}
  

  
RNBoolean R3SurfelNode::
HasActive(void) const
{
  // Return whether node has active surfels 
  if (!flags[R3_SURFEL_NODE_FLAGS_UPTODATE_FLAG])
    ((R3SurfelNode *) this)->UpdateFlags();
  return flags[R3_SURFEL_NODE_HAS_ACTIVE_FLAG];
}



RNBoolean R3SurfelNode::
HasNormals(void) const
{
  // Return whether node has surfels with normals
  if (!flags[R3_SURFEL_NODE_FLAGS_UPTODATE_FLAG])
    ((R3SurfelNode *) this)->UpdateFlags();
  return flags[R3_SURFEL_NODE_HAS_NORMALS_FLAG];
}



RNBoolean R3SurfelNode::
HasAerial(void) const
{
  // Return whether node has surfels collected with aerial scanner
  if (!flags[R3_SURFEL_NODE_FLAGS_UPTODATE_FLAG])
    ((R3SurfelNode *) this)->UpdateFlags();
  return flags[R3_SURFEL_NODE_HAS_AERIAL_FLAG];
}



RNBoolean R3SurfelNode::
HasTerrestrial(void) const
{
  // Return whether node has surfels collected with terrestrial scanner
  if (!flags[R3_SURFEL_NODE_FLAGS_UPTODATE_FLAG]) 
    ((R3SurfelNode *) this)->UpdateFlags();
  return flags[R3_SURFEL_NODE_HAS_TERRESTRIAL_FLAG];
}



int R3SurfelNode::
TreeLevel(void) const
{
  // Return level in part tree (root is 0)
  int level = 0;
  R3SurfelNode *ancestor = parent;
  while (ancestor) { level++; ancestor = ancestor->parent; }
  return level;
}



////////////////////////////////////////////////////////////////////////
// ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelObject *R3SurfelNode::
Object(RNBoolean search_ancestors, RNBoolean search_descendants) const
{
  // Return object, if there is one associated with this node
  if (object) return object;

  // Search descendants for object
  if (search_descendants) {
    RNScalar best_complexity = 0;
    R3SurfelObject *best_object = NULL;
    for (int i = 0; i < NParts(); i++) {
      R3SurfelNode *part = Part(i);
      R3SurfelObject *object = part->Object(FALSE, TRUE);
      if (object) {
        RNScalar complexity = object->Complexity();
        if (complexity > best_complexity) {
          best_complexity = complexity;
          best_object = object;
        }
      }
    }
    if (best_object) return best_object;
  }

  // Search ancestors for object
  if (search_ancestors) {
    const R3SurfelNode *node = this;
    while (node) {
      R3SurfelObject *object = node->Object(FALSE, FALSE);
      if (object) return object;
      node = node->Parent();
    }
  }

  // Object not found
  return NULL;
}



R3SurfelScan *R3SurfelNode::
Scan(RNBoolean search_ancestors) const
{
  // Check if should search ancestors
  if (!search_ancestors) return scan;

  // Search ancestors for scan
  const R3SurfelNode *node = this;
  while (node) {
    R3SurfelScan *scan = node->Scan(FALSE);
    if (scan) return scan;
    node = node->Parent();
  }

  // Scan not found
  return NULL;
}



R3SurfelPointSet *R3SurfelNode::
PointSet(RNBoolean leaf_level, RNScalar max_resolution) const
{
  // Allocate point set
  R3SurfelPointSet *pointset = new R3SurfelPointSet();
  if (!pointset) {
    RNFail("Unable to allocate point set\n");
    return NULL;
  }

  // Insert surfels from this node into pointset
  InsertIntoPointSet(pointset, leaf_level, max_resolution);

  // Return pointset
  return pointset;
}


  
void R3SurfelNode::
InsertIntoPointSet(R3SurfelPointSet *pointset, RNBoolean leaf_level, RNScalar max_resolution) const
{
  // Insert points
  if (leaf_level && (NParts() > 0)) {
    // Insert points from all decendent parts at leaf level
    for (int i = 0; i < NParts(); i++) {
      R3SurfelNode *part = Part(i);
      part->InsertIntoPointSet(pointset, leaf_level, max_resolution);
    }
  }
  else {
    // Insert points from all blocks in this node
    for (int i = 0; i < NBlocks(); i++) {
      R3SurfelBlock *block = Block(i);
      pointset->InsertPoints(block, max_resolution);
    }
  }
}



////////////////////////////////////////////////////////////////////////
// PROPERTY MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
SetName(const char *name)
{
  // Delete previous name
  if (this->name) free(this->name);
  this->name = (name) ? RNStrdup(name) : NULL;

  // Mark scene as dirty
  if (Tree() && Tree()->Scene()) Tree()->Scene()->SetDirty();
}



void R3SurfelNode::
SetData(void *data) 
{
  // Set user data
  this->data = data;
}



////////////////////////////////////////////////////////////////////////
// HIERARCHY MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
SetParent(R3SurfelNode *parent)
{
  // Just checking
  assert(parent);
  assert(this->parent);
  assert(tree);
  assert(tree == this->parent->tree);
  assert(tree == parent->tree);
  if (parent == this->parent) return;

  // Find least common ancestor
  R3SurfelNode *least_common_ancestor = NULL;

  // Invalidate properties starting at current parent
  R3SurfelNode *ancestor = this->parent;
  while (ancestor && (ancestor != least_common_ancestor)) {
    if (ancestor->complexity == -1) break;
    ancestor->complexity = -1;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor->timestamp_range[0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Invalidate properties starting at new parent
  ancestor = parent;
  while (ancestor && (ancestor != least_common_ancestor)) {
    if (ancestor->complexity == -1) break;
    ancestor->complexity = -1;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor->timestamp_range[0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Update hierarchy
  this->parent->parts.Remove(this);
  parent->parts.Insert(this);
  this->parent = parent;

  // Mark scene as dirty
  if (Tree() && Tree()->Scene()) Tree()->Scene()->SetDirty();
}



////////////////////////////////////////////////////////////////////////
// BLOCK MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
InsertBlock(R3SurfelBlock *block)
{
  // Just checking
  assert(block->node == NULL);
  assert(!blocks.FindEntry(block));

  // Insert block
  blocks.Insert(block);

  // Update properties
  R3SurfelNode *ancestor = this;
  while (ancestor) {
    if (ancestor->complexity == -1) break;
    ancestor->complexity = -1;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor->timestamp_range[0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Mark resolution out of date
  resolution = -1;

  // Update block
  block->UpdateAfterInsert(this);

  // Update object
  if (object) object->UpdateAfterInsertBlock(this, block);

  // Mark scene as dirty
  if (Tree() && Tree()->Scene()) Tree()->Scene()->SetDirty();
}



void R3SurfelNode::
RemoveBlock(R3SurfelBlock *block)
{
  // Just checking
  assert(block->node == this);
  assert(blocks.FindEntry(block));

  // Update object
  if (object) object->UpdateBeforeRemoveBlock(this, block);

  // Update block
  block->UpdateBeforeRemove(this);

  // Remove block
  blocks.Remove(block);

  // Mark resolution out of date
  resolution = -1;

  // Invalidate properties
  R3SurfelNode *ancestor = this;
  while (ancestor) {
    if (ancestor->complexity == -1) break;
    ancestor->complexity = -1;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor->timestamp_range[0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Mark scene as dirty
  if (Tree() && Tree()->Scene()) Tree()->Scene()->SetDirty();
}



///////////////////////////////////////////////////////////////////////
// MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
Transform(const R3Affine& transformation) 
{
  // Transform all blocks
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    block->Transform(transformation);
  }

  // Update resolution
  RNScalar scale = transformation.ScaleFactor();
  if ((resolution > 0) && (RNIsNotEqual(scale, 1.0))) resolution /= scale * scale;

  // Invalidate bounding boxes starting at this node
  R3SurfelNode *ancestor = this;
  while (ancestor) {
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor->timestamp_range[0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Update object
  if (object) object->UpdateAfterTransform(this);

  // Mark scene as dirty
  if (Tree() && Tree()->Scene()) Tree()->Scene()->SetDirty();
}



void R3SurfelNode::
SetMarks(RNBoolean mark)
{
  // Set mark for all blocks
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    block->SetMarks(mark);
  }
}



///////////////////////////////////////////////////////////////////////
// MEMORY MANGEMENT FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
ReadBlocks(RNBoolean entire_subtree)
{
  // Read blocks in node
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    R3SurfelDatabase *database = block->Database();
    if (database) database->ReadBlock(block);
  }

  // Read blocks in parts
  if (entire_subtree) {
    for (int i = 0; i < NParts(); i++) {
      R3SurfelNode *part = Part(i);
      part->ReadBlocks(entire_subtree);
    }
  }
}



void R3SurfelNode::
ReleaseBlocks(RNBoolean entire_subtree)
{
  // Release blocks in node
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    R3SurfelDatabase *database = block->Database();
    if (database) database->ReleaseBlock(block);
  }

  // Release blocks in parts
  if (entire_subtree) {
    for (int i = 0; i < NParts(); i++) {
      R3SurfelNode *part = Part(i);
      part->ReleaseBlocks(entire_subtree);
    }
  }
}



RNBoolean R3SurfelNode::
AreBlocksResident(void) const
{
  // Return whether blocks are in memory
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    R3SurfelDatabase *database = block->Database();
    if (!database->IsBlockResident(block)) return FALSE;
  }

  // All blocks are resident
  return TRUE;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print node
  if (prefix) fprintf(fp, "%s", prefix);
  fprintf(fp, "%d %s", TreeIndex(), (Name()) ? Name() : "-");
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");

  // Print all surfel blocks
  if (prefix) fprintf(fp, "%s", prefix);
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    fprintf(fp, "%d ", block->DatabaseIndex());
  }
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}




void R3SurfelNode::
Draw(RNFlags flags) const
{
  // Draw all blocks
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    block->Draw(flags);
  }
}



int R3SurfelNode::
DrawResidentDescendents(RNFlags draw_flags) const
{
  // Initialize return status
  int status = 0;

  // Draw parts
  if (NParts() > 0) {
    // Draw resident descendents of all parts
    status = 1;
    for (int i = 0; i < NParts(); i++) {
      R3SurfelNode *part = Part(i);
      status &= part->DrawResidentDescendents(draw_flags);
    }
  }

  // Draw node if resident
  if (status == 0) {
    if (AreBlocksResident()) {
      Draw(draw_flags);
      status = 1;
    }
  }

  // Return whether drew everything
  return status;
}



int R3SurfelNode::
DrawResidentAncestor(RNFlags draw_flags) const
{
#if 1
  // Draw K*K*K points in bounding box
  int K = 3;
  RNGrfxBegin(RN_GRFX_POINTS);
  for (int i = 0; i < K; i++) {
    double x = bbox.XMin() + bbox.XLength() * (i+0.5)/K;
    for (int j = 0; j < K; j++) {
      double y = bbox.YMin() + bbox.YLength() * (j+0.5)/K;
      for (int k = 0; k < K; k++) {
        double z = bbox.ZMin() + bbox.ZLength() * (k+0.5)/K;
        R3LoadPoint(x, y, z);
      }
    }
  }
  RNGrfxEnd();
#else
  // Find closest ancestor that is resident 
  const R3SurfelNode *ancestor = this;
  while (ancestor) {
    if (ancestor->AreBlocksResident()) break;
    ancestor = ancestor->Parent();
  }
      
  // Check ancestor
  if (!ancestor) return 0;

  // Set clip planes
  if (ancestor != this) {
    // Set up clip plane normals
    static GLdouble lowx_plane[4] =  {  1,  0,  0,  0 };
    static GLdouble lowy_plane[4] =  {  0,  1,  0,  0 };
    static GLdouble lowz_plane[4] =  {  0,  0,  1,  0 };
    static GLdouble highx_plane[4] = { -1,  0,  0,  0 };
    static GLdouble highy_plane[4] = {  0, -1,  0,  0 };
    static GLdouble highz_plane[4] = {  0,  0, -1,  0 };
        
    // Set up clip plane coordinates based on node bounding box
    const R3Box &b = BBox();
    lowx_plane[3] = -b.XMin();
    lowy_plane[3] = -b.YMin();
    lowz_plane[3] = -b.ZMin();
    highx_plane[3] = b.XMax();
    highy_plane[3] = b.YMax();
    highz_plane[3] = b.ZMax();
        
    // Send clip planes to OpenGL
    glClipPlane(GL_CLIP_PLANE0, lowx_plane);
    glClipPlane(GL_CLIP_PLANE1, lowy_plane);
    glClipPlane(GL_CLIP_PLANE2, lowz_plane);
    glClipPlane(GL_CLIP_PLANE3, highx_plane);
    glClipPlane(GL_CLIP_PLANE4, highy_plane);
    glClipPlane(GL_CLIP_PLANE5, highz_plane);
        
    // Enable clip planes
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);
    glEnable(GL_CLIP_PLANE2);
    glEnable(GL_CLIP_PLANE3);
    glEnable(GL_CLIP_PLANE4);
    glEnable(GL_CLIP_PLANE5);
  }
      
  // Draw ancestor node
  ancestor->Draw(draw_flags);

  // Disable clip planes
  if (ancestor != this) {
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
    glDisable(GL_CLIP_PLANE3);
    glDisable(GL_CLIP_PLANE4);
    glDisable(GL_CLIP_PLANE5);
  }
#endif

  // Return success
  return 1;
}
  
  
  
////////////////////////////////////////////////////////////////////////
// STRUCTURE UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
UpdateAfterInsert(R3SurfelTree *tree)
{
  // Invalidate properties
  R3SurfelNode *ancestor = parent;
  while (ancestor) {
    if (ancestor->complexity == -1) break;
    ancestor->complexity = -1;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor->timestamp_range[0] = FLT_MAX;
    // while (ancestor->NBlocks()) { 
    //   R3SurfelBlock *block = ancestor->Block(0);
    //   ancestor->RemoveBlock(block);
    //   database->RemoveBlock(block);
    //   delete block;
    // }
    ancestor = ancestor->parent;
  }
}



void R3SurfelNode::
UpdateBeforeRemove(R3SurfelTree *tree)
{
  // Invalidate properties
  R3SurfelNode *ancestor = parent;
  while (ancestor) {
    if (ancestor->complexity == -1) break;
    ancestor->complexity = -1;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor->timestamp_range[0] = FLT_MAX;
    // while (ancestor->NBlocks()) { 
    //   R3SurfelBlock *block = ancestor->Block(0);
    //   ancestor->RemoveBlock(block);
    //   database->RemoveBlock(block);
    //   delete block;
    // }
    ancestor = ancestor->parent;
  }
}



////////////////////////////////////////////////////////////////////////
// PROPERTY UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
UpdateProperties(void)
{
  // Update surfels
  UpdateSurfelNormals();

  // Update properties
  double dummy = 0;
  dummy += Complexity();
  dummy += Resolution();
  dummy += BBox().Min().X();
  dummy += ElevationRange().Min();
  dummy += TimestampRange().Min();
  if (dummy == 927612.21242) {
    printf("Amazing!\n");
  }
}



void R3SurfelNode::
UpdateBBox(void)
{
  // Check if bounding box is uptodate
  if (bbox[0][0] != FLT_MAX) return;

  // Initialize bounding box
  bbox = R3null_box;

  // Union bounding boxes of blocks
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    bbox.Union(block->BBox());
  }

  // Union bounding boxes of parts
  for (int i = 0; i < NParts(); i++) {
    R3SurfelNode *part = Part(i);
    bbox.Union(part->BBox());
  }
}



void R3SurfelNode::
UpdateElevationRange(void)
{
  // Check if range is uptodate
  if (elevation_range[0] != FLT_MAX) return;

  // Initialize elevation range
  elevation_range.Reset(FLT_MAX,-FLT_MAX);

  // Compute elevation range
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    RNInterval block_range = block->ElevationRange();
    elevation_range.Union(block_range);
  }
}



void R3SurfelNode::
UpdateTimestampRange(void)
{
  // Check if range is uptodate
  if (timestamp_range[0] != FLT_MAX) return;

  // Initialize timestamp range
  timestamp_range.Reset(FLT_MAX,-FLT_MAX);

  // Union timestamp ranges of blocks
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    timestamp_range.Union(block->TimestampRange());
  }

  // Union timestamp ranges of parts
  for (int i = 0; i < NParts(); i++) {
    R3SurfelNode *part = Part(i);
    timestamp_range.Union(part->TimestampRange());
  }
}



void R3SurfelNode::
UpdateComplexity(void)
{
  // Check if complexity is uptodate
  if (complexity >= 0) return;

  // Sum complexity of blocks
  complexity = 0;
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    complexity += block->NSurfels();
  }
}



void R3SurfelNode::
UpdateResolution(void)
{
  // Check if resolution is uptodate
  if (resolution >= 0) return;

  // Initialize resolution
  resolution = 0;

  // Compute minimum of block resolutions
  if (Complexity() > 0) {
    resolution = FLT_MAX;
    assert(NBlocks() > 0);
    for (int i = 0; i < NBlocks(); i++) {
      R3SurfelBlock *block = Block(i);
      RNScalar block_resolution = block->Resolution();
      if (block_resolution < resolution) {
        resolution = block_resolution;
      }
    }
  }
}



void R3SurfelNode::
UpdateFlags(void)
{
  // Reset flags
  flags.Remove(R3_SURFEL_NODE_HAS_ACTIVE_FLAG);
  flags.Remove(R3_SURFEL_NODE_HAS_NORMALS_FLAG);
  flags.Remove(R3_SURFEL_NODE_HAS_AERIAL_FLAG);
  flags.Remove(R3_SURFEL_NODE_HAS_TERRESTRIAL_FLAG);

  // Update flags
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    if (block->HasActive()) flags.Add(R3_SURFEL_NODE_HAS_ACTIVE_FLAG);
    if (block->HasNormals()) flags.Add(R3_SURFEL_NODE_HAS_NORMALS_FLAG);
    if (block->HasAerial()) flags.Add(R3_SURFEL_NODE_HAS_AERIAL_FLAG);
    if (block->HasTerrestrial()) flags.Add(R3_SURFEL_NODE_HAS_TERRESTRIAL_FLAG);
  }

  // Mark flags uptodate
  flags.Add(R3_SURFEL_NODE_FLAGS_UPTODATE_FLAG);
}



////////////////////////////////////////////////////////////////////////
// SURFEL UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNode::
UpdateSurfelNormals(void) 
{
#if 0
  // Create pointset
  R3SurfelPointSet pointset;
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    pointset.InsertPoints(block);
  }

  // Update normals
  pointset.UpdateNormals();
#else
  // Update normals block-by-block
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    block->UpdateSurfelNormals();
  }
#endif
}



} // namespace gaps
