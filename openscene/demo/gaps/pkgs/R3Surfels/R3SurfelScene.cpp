/* Source file for the R3 surfel scene class */



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

#define R3_SURFEL_SCENE_DIRTY_FLAG                  0x01



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelScene::
R3SurfelScene(const char *name)
  : tree(NULL),
    objects(),
    labels(),
    object_properties(),
    label_properties(),
    object_relationships(),
    label_relationships(),
    assignments(),
    scans(),
    images(),
    features(),
    transformation(R4Matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1), 0),
    filename(NULL),
    rwaccess(NULL),
    name((name) ? RNStrdup(name) : NULL),
    comments(),
    flags(R3_SURFEL_SCENE_DIRTY_FLAG)
{
  // Create tree
  tree = new R3SurfelTree();
  tree->scene = this;

  // Create root object
  R3SurfelObject *object = new R3SurfelObject("Root");
  InsertObject(object, NULL);

  // Create root label
  R3SurfelLabel *label = new R3SurfelLabel("Root");
  InsertLabel(label, NULL);
}



R3SurfelScene::
R3SurfelScene(const R3SurfelScene& scene)
{
  // Copy everything
  RNAbort("Not implemented");
}



R3SurfelScene::
~R3SurfelScene(void)
{
  // Delete everything
  while (NFeatures() > 0) delete Feature(NFeatures()-1);
  while (NLabelAssignments() > 0) delete LabelAssignment(NLabelAssignments()-1);
  while (NObjectRelationships() > 0) delete ObjectRelationship(NObjectRelationships()-1);
  while (NLabelRelationships() > 0) delete LabelRelationship(NLabelRelationships()-1);
  while (NObjectProperties() > 0) delete ObjectProperty(NObjectProperties()-1);
  while (NLabelProperties() > 0) delete LabelProperty(NLabelProperties()-1);
  while (NLabels() > 0) delete Label(NLabels()-1);
  while (NObjects() > 0) delete Object(NObjects()-1);
  while (NScans() > 0) delete Scan(NScans()-1);
  while (NImages() > 0) delete Image(NImages()-1);
  
  // Delete tree
  if (tree) delete tree;
  
  // Delete filename
  if (filename) free(filename);

  // Delete rwaccess
  if (rwaccess) free(rwaccess);

  // Delete name
  if (name) free(name);

  // Delete comments
  for (int i = 0; i < comments.NEntries(); i++) free(comments[i]);
}



////////////////////////////////////////////////////////////////////////
// ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelObject *R3SurfelScene::
FindObjectByName(const char *object_name) const
{
  // Search for object
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    if (!object->Name()) continue;
    if (strcmp(object->Name(), object_name)) continue;
    return object;
  }

  // Not found
  return NULL;
}



R3SurfelObject *R3SurfelScene::
FindObjectByIdentifier(int identifier) const
{
  // Search for object
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    if (object->Identifier() < 0) continue;
    if (object->Identifier() != identifier) continue;
    return object;
  }

  // Not found
  return NULL;
}



R3SurfelLabel *R3SurfelScene::
FindLabelByName(const char *label_name) const
{
  // Search for label
  for (int i = 0; i < NLabels(); i++) {
    R3SurfelLabel *label = Label(i);
    if (!label->Name()) continue;
    if (strcmp(label->Name(), label_name)) continue;
    return label;
  }

  // Not found
  return NULL;
}



R3SurfelLabel *R3SurfelScene::
FindLabelByIdentifier(int identifier) const
{
  // Search for label
  for (int i = 0; i < NLabels(); i++) {
    R3SurfelLabel *label = Label(i);
    if (label->Identifier() < 0) continue;
    if (label->Identifier() != identifier) continue;
    return label;
  }

  // Not found
  return NULL;
}



R3SurfelLabel *R3SurfelScene::
FindLabelByAssignmentKeystroke(int key) const
{
  // Search for label
  for (int i = 0; i < NLabels(); i++) {
    R3SurfelLabel *label = Label(i);
    if (label->AssignmentKeystroke() < 0) continue;
    if (label->AssignmentKeystroke() != key) continue;
    return label;
  }

  // Not found
  return NULL;
}



R3SurfelLabelAssignment *R3SurfelScene::
FindLabelAssignment(R3SurfelObject *object, R3SurfelLabel *label, RNScalar confidence, int originator) const
{
  // Search for label assignment
  for (int i = 0; i < object->NLabelAssignments(); i++) {
    R3SurfelLabelAssignment *assignment = object->LabelAssignment(i);
    if (assignment->Label() != label) continue;
    if (assignment->Confidence() != confidence) continue;
    if (assignment->Originator() != originator) continue;
    return assignment;
  }

  // Not found
  return NULL;
}



R3SurfelScan *R3SurfelScene::
FindScanByName(const char *scan_name) const
{
  // Search for scan
  for (int i = 0; i < NScans(); i++) {
    R3SurfelScan *scan = Scan(i);
    if (!scan->Name()) continue;
    if (strcmp(scan->Name(), scan_name)) continue;
    return scan;
  }

  // Not found
  return NULL;
}



R3SurfelImage *R3SurfelScene::
FindImageByName(const char *image_name) const
{
  // Search for image
  for (int i = 0; i < NImages(); i++) {
    R3SurfelImage *image = Image(i);
    if (!image->Name()) continue;
    if (strcmp(image->Name(), image_name)) continue;
    return image;
  }

  // Not found
  return NULL;
}



R3SurfelImage *R3SurfelScene::
FindImageByBestView(const R3Point& query_position, const R3Vector& query_normal)
{
  // Search for image with best view of a point
  RNScalar best_score = 0;
  R3SurfelImage *best_image = NULL;
  for (int i = 0; i < NImages(); i++) {
    R3SurfelImage *image = Image(i);

    // Get/check point depth
    R3Point camera_position = image->TransformFromWorldToCamera(query_position);
    RNScalar point_depth = -camera_position.Z();
    if (RNIsNegativeOrZero(point_depth)) continue;

    // Get/check projection into image
    R2Point image_position = image->TransformFromCameraToImage(camera_position);
    int ix = (int) (image_position.X() + 0.5);
    if ((ix < 0) || (ix >= image->ImageWidth())) continue;
    int iy = (int) (image_position.Y() + 0.5);
    if ((iy < 0) || (iy >= image->ImageHeight())) continue;

#if 0
    // Get/check image depth
    const R2Grid *depth_channel = image->DepthChannel();
    if (depth_channel) {
      RNScalar image_depth = depth_channel->GridValue(ix, iy);
      if (image_depth != R2_GRID_UNKNOWN_VALUE) {
        RNScalar delta_depth = fabs(image_depth - point_depth);
        if (delta_depth > 0.1 * point_depth) continue;
      }
    }
#endif
   
#if 0
    // Get/check NdotV
    RNScalar NdotV = 1.0;
    if (!query_normal.IsZero()) {
      R3Vector V = image->Viewpoint() - query_position;
      V.Normalize();
      RNScalar NdotV = query_normal.Dot(V);
      if (RNIsNegativeOrZero(NdotV)) NdotV = continue;
    }
#endif
   
    // Get image centrality
    double dx = fabs(ix - 0.5*image->ImageWidth()) / (0.5*image->ImageWidth());
    double dy = fabs(iy - 0.5*image->ImageHeight()) / (0.5*image->ImageHeight());
    double image_centrality = (1.0 - dx) * (1.0 - dy);

    // Get distance
    RNScalar dd = R3SquaredDistance(image->Viewpoint(), query_position);

    // Get/check score
    RNScalar score = image_centrality / dd;
    if (score > best_score) {
      best_image = image;
      best_score = score;
    }
  }

  // Return best image
  return best_image;
}



R3SurfelFeature *R3SurfelScene::
FindFeatureByName(const char *feature_name) const
{
  // Search for feature
  for (int i = 0; i < NFeatures(); i++) {
    R3SurfelFeature *feature = Feature(i);
    if (!feature->Name()) continue;
    if (strcmp(feature->Name(), feature_name)) continue;
    return feature;
  }

  // Not found
  return NULL;
}



////////////////////////////////////////////////////////////////////////
// PROPERTY MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelScene::
SetName(const char *name)
{
  // Delete previous name
  if (this->name) free(this->name);

  // Set new name
  this->name = (name) ? RNStrdup(name) : NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
SetTransformation(const R3Affine& transformation, RNBoolean update_surfels)
{
  // Set transformation (does not change surfels)
  this->transformation = transformation;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
Transform(const R3Affine& transformation, RNBoolean update_surfels)
{
  // Check if should update surfels
  if (update_surfels) {
    // Compute total transformation
    R3Affine t = R3identity_affine;
    t.Transform(transformation);
    t.Transform(this->transformation);
    this->transformation = R3identity_affine;
    
    // Transform all nodes                                                                                                     
    R3SurfelTree *tree = Tree();
    for (int i = 0; i < tree->NNodes(); i++) {
      R3SurfelNode *node = tree->Node(i);
      node->Transform(t);
    }

    // Transform all scans                                                                                                
    for (int i = 0; i < NScans(); i++) {
      R3SurfelScan *scan = Scan(i);
      scan->Transform(t);
    }

    // Transform all image poses                                                                                               
    for (int i = 0; i < NImages(); i++) {
      R3SurfelImage *image = Image(i);
      image->Transform(t);
    }
  }
  else {
    // Update transformation only
    R3Affine t = R3identity_affine;
    t.Transform(transformation);
    t.Transform(this->transformation);
    this->transformation = t;
  }

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



////////////////////////////////////////////////////////////////////////
// STRUCTURE MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelScene::
InsertObject(R3SurfelObject *object, R3SurfelObject *parent)
{
  // Just checking
  assert(object);
  assert(object->scene == NULL);
  assert(object->scene_index == -1);

  // Insert object into scene
  object->scene = this;
  object->scene_index = objects.NEntries();
  objects.Insert(object);

  // Insert object into parent
  object->parent = parent;
  if (parent) parent->parts.Insert(object);

  // Update object
  object->UpdateAfterInsert(this);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
MergeObject(R3SurfelObject *dst_object, R3SurfelObject *src_object)
{
  // Just checking
  assert(dst_object && src_object);
  assert(dst_object->scene == this);
  assert(dst_object->scene_index >= 0);
  assert(src_object->scene == this);
  assert(src_object->scene_index >= 0);

  // Move all nodes from src_object into dst_object
  while (src_object->NNodes() > 0) {
    R3SurfelNode *node = src_object->Node(src_object->NNodes()-1);
    src_object->RemoveNode(node);
    dst_object->InsertNode(node);
  }

  // Move all parts from src_object into dst_object
  while (src_object->NParts() > 0) {
    R3SurfelObject *part = src_object->Part(src_object->NParts()-1);
    RemoveObject(part);
    InsertObject(part, dst_object);
  }

  // Move all object properties from src_object into dst_object
  while (src_object->NObjectProperties() > 0) {
    R3SurfelObjectProperty *property = src_object->ObjectProperty(src_object->NObjectProperties()-1);
    src_object->UpdateBeforeRemoveObjectProperty(property);
    property->object = dst_object;
    dst_object->UpdateAfterInsertObjectProperty(property);
  }

  // Move all object relationships from src_object into dst_object
  while (src_object->NObjectRelationships() > 0) {
    R3SurfelObjectRelationship *relationship = src_object->ObjectRelationship(src_object->NObjectRelationships()-1);
    src_object->UpdateBeforeRemoveObjectRelationship(relationship);
    for (int i = 0; i < relationship->objects.NEntries(); i++) {
      if (relationship->objects.Kth(i) == src_object) {
        RNArrayEntry *entry = relationship->objects.KthEntry(i);
        relationship->objects.EntryContents(entry) = dst_object;
        break;
      }
    }
    dst_object->UpdateAfterInsertObjectRelationship(relationship);
  }

  // Move all label assignments from src_object into dst_object
  while (src_object->NLabelAssignments() > 0) {
    R3SurfelLabelAssignment *assignment = src_object->LabelAssignment(src_object->NLabelAssignments()-1);
    src_object->UpdateBeforeRemoveLabelAssignment(assignment);
    assignment->object = dst_object;
    dst_object->UpdateAfterInsertLabelAssignment(assignment);
  }

  // Remove src_object
  RemoveObject(src_object);

  // Delete src_object
  delete src_object;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveObject(R3SurfelObject *object)
{
  // Just checking
  assert(object);
  assert(object->scene == this);
  assert(object->scene_index >= 0);
  assert(objects.Kth(object->scene_index) == object);

  // Remove object propertys from scene
  while (object->NObjectProperties() > 0) {
    R3SurfelObjectProperty *property = object->ObjectProperty(object->NObjectProperties()-1);
    RemoveObjectProperty(property);
  }

  // Remove object relationships from scene
  while (object->NObjectRelationships() > 0) {
    R3SurfelObjectRelationship *relationship = object->ObjectRelationship(object->NObjectRelationships()-1);
    RemoveObjectRelationship(relationship);
  }

  // Remove label assignments
  while (object->NLabelAssignments() > 0) {
    R3SurfelLabelAssignment *assignment = object->LabelAssignment(object->NLabelAssignments()-1);
    RemoveLabelAssignment(assignment);
  }

  // Update object
  object->UpdateBeforeRemove(this);

  // Remove object from parent
  if (object->parent) {
    object->parent->parts.Remove(object);
    object->parent = NULL;
  }

  // Remove object from parts
  for (int i = 0; i < object->NParts(); i++) {
    R3SurfelObject *part = object->Part(i);
    part->parent = NULL;
  }

  // Remove parts from object
  object->parts.Empty();

  // Remove object from scene
  RNArrayEntry *entry = objects.KthEntry(object->scene_index);
  R3SurfelObject *tail = objects.Tail();
  tail->scene_index = object->scene_index;
  objects.EntryContents(entry) = tail;
  objects.RemoveTail();
  object->scene_index = -1;
  object->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertLabel(R3SurfelLabel *label, R3SurfelLabel *parent)
{
  // Just checking
  assert(label);
  assert(label->scene == NULL);
  assert(label->scene_index == -1);

  // Insert label 
  label->scene = this;
  label->scene_index = labels.NEntries();
  labels.Insert(label);

  // Insert label into parent
  label->parent = parent;
  if (parent) parent->parts.Insert(label);

  // Update label
  label->UpdateAfterInsert(this);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveLabel(R3SurfelLabel *label)
{
  // Just checking
  assert(label);
  assert(label->scene == this);
  assert(label->scene_index >= 0);
  assert(labels.Kth(label->scene_index) == label);

  // Remove label properties from scene
  while (label->NLabelProperties() > 0) {
    R3SurfelLabelProperty *property = label->LabelProperty(label->NLabelProperties()-1);
    RemoveLabelProperty(property);
  }

  // Remove label relationships from scene
  while (label->NLabelRelationships() > 0) {
    R3SurfelLabelRelationship *relationship = label->LabelRelationship(label->NLabelRelationships()-1);
    RemoveLabelRelationship(relationship);
  }

  // Remove label assignments
  while (label->NLabelAssignments() > 0) {
    R3SurfelLabelAssignment *assignment = label->LabelAssignment(label->NLabelAssignments()-1);
    RemoveLabelAssignment(assignment);
  }

  // Update label
  label->UpdateBeforeRemove(this);

  // Remove label from parent
  if (label->parent) {
    label->parent->parts.Remove(label);
    label->parent = NULL;
  }

  // Remove label from parts
  for (int i = 0; i < label->NParts(); i++) {
    R3SurfelLabel *part = label->Part(i);
    part->parent = NULL;
  }

  // Remove parts from label
  label->parts.Empty();

  // Remove label from scene
  RNArrayEntry *entry = labels.KthEntry(label->scene_index);
  R3SurfelLabel *tail = labels.Tail();
  tail->scene_index = label->scene_index;
  labels.EntryContents(entry) = tail;
  labels.RemoveTail();
  label->scene_index = -1;
  label->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertObjectProperty(R3SurfelObjectProperty *property)
{
  // Just checking
  assert(property);
  assert(property->scene == NULL);
  assert(property->scene_index == -1);

  // Insert property into scene
  property->scene = this;
  property->scene_index = object_properties.NEntries();
  object_properties.Insert(property);

  // Update object
  R3SurfelObject *object = property->Object();
  object->UpdateAfterInsertObjectProperty(property);

  // Update object property
  property->UpdateAfterInsert(this);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveObjectProperty(R3SurfelObjectProperty *property)
{
  // Just checking
  assert(property);
  assert(property->scene == this);
  assert(property->scene_index >= 0);
  assert(object_properties.Kth(property->scene_index) == property);

  // Update object property
  property->UpdateBeforeRemove(this);

  // Update object
  R3SurfelObject *object = property->Object();
  object->UpdateBeforeRemoveObjectProperty(property);

  // Remove property from scene
  RNArrayEntry *entry = object_properties.KthEntry(property->scene_index);
  R3SurfelObjectProperty *tail = object_properties.Tail();
  tail->scene_index = property->scene_index;
  object_properties.EntryContents(entry) = tail;
  object_properties.RemoveTail();
  property->scene_index = -1;
  property->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertLabelProperty(R3SurfelLabelProperty *property)
{
  // Just checking
  assert(property);
  assert(property->scene == this);
  assert(property->scene_index == -1);

  // Insert property into scene
  property->scene = this;
  property->scene_index = label_properties.NEntries();
  label_properties.Insert(property);

  // Update label
  R3SurfelLabel *label = property->Label();
  label->UpdateAfterInsertLabelProperty(property);

  // Update label property
  property->UpdateAfterInsert(this);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveLabelProperty(R3SurfelLabelProperty *property)
{
  // Just checking
  assert(property);
  assert(property->scene == this);
  assert(property->scene_index >= 0);
  assert(label_properties.Kth(property->scene_index) == property);

  // Update label property
  property->UpdateAfterInsert(this);

  // Update label
  R3SurfelLabel *label = property->Label();
  label->UpdateBeforeRemoveLabelProperty(property);

  // Remove property from scene
  RNArrayEntry *entry = label_properties.KthEntry(property->scene_index);
  R3SurfelLabelProperty *tail = label_properties.Tail();
  tail->scene_index = property->scene_index;
  label_properties.EntryContents(entry) = tail;
  label_properties.RemoveTail();
  property->scene_index = -1;
  property->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertObjectRelationship(R3SurfelObjectRelationship *relationship)
{
  // Just checking
  assert(relationship);
  assert(relationship->scene == NULL);
  assert(relationship->scene_index == -1);

  // Insert relationship into scene
  relationship->scene = this;
  relationship->scene_index = object_relationships.NEntries();
  object_relationships.Insert(relationship);

  // Update objects
  for (int i = 0; i < relationship->NObjects(); i++) {
    R3SurfelObject *object = relationship->Object(i);
    object->UpdateAfterInsertObjectRelationship(relationship);
  }

  // Update object relationship
  relationship->UpdateAfterInsert(this);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveObjectRelationship(R3SurfelObjectRelationship *relationship)
{
  // Just checking
  assert(relationship);
  assert(relationship->scene == this);
  assert(relationship->scene_index >= 0);
  assert(object_relationships.Kth(relationship->scene_index) == relationship);

  // Update object relationship
  relationship->UpdateBeforeRemove(this);

  // Update objects
  for (int i = 0; i < relationship->NObjects(); i++) {
    R3SurfelObject *object = relationship->Object(i);
    object->UpdateBeforeRemoveObjectRelationship(relationship);
  }

  // Remove relationship from scene
  RNArrayEntry *entry = object_relationships.KthEntry(relationship->scene_index);
  R3SurfelObjectRelationship *tail = object_relationships.Tail();
  tail->scene_index = relationship->scene_index;
  object_relationships.EntryContents(entry) = tail;
  object_relationships.RemoveTail();
  relationship->scene_index = -1;
  relationship->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertLabelRelationship(R3SurfelLabelRelationship *relationship)
{
  // Just checking
  assert(relationship);
  assert(relationship->scene == this);
  assert(relationship->scene_index == -1);

  // Insert relationship into scene
  relationship->scene = this;
  relationship->scene_index = label_relationships.NEntries();
  label_relationships.Insert(relationship);

  // Update labels
  for (int i = 0; i < relationship->NLabels(); i++) {
    R3SurfelLabel *label = relationship->Label(i);
    label->UpdateAfterInsertLabelRelationship(relationship);
  }

  // Update label relationship
  relationship->UpdateAfterInsert(this);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveLabelRelationship(R3SurfelLabelRelationship *relationship)
{
  // Just checking
  assert(relationship);
  assert(relationship->scene == this);
  assert(relationship->scene_index >= 0);
  assert(label_relationships.Kth(relationship->scene_index) == relationship);

  // Update label relationship
  relationship->UpdateAfterInsert(this);

  // Update labels
  for (int i = 0; i < relationship->NLabels(); i++) {
    R3SurfelLabel *label = relationship->Label(i);
    label->UpdateBeforeRemoveLabelRelationship(relationship);
  }

  // Remove relationship from scene
  RNArrayEntry *entry = label_relationships.KthEntry(relationship->scene_index);
  R3SurfelLabelRelationship *tail = label_relationships.Tail();
  tail->scene_index = relationship->scene_index;
  label_relationships.EntryContents(entry) = tail;
  label_relationships.RemoveTail();
  relationship->scene_index = -1;
  relationship->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertLabelAssignment(R3SurfelLabelAssignment *assignment)
{
  // Just checking
  assert(assignment);
  assert(assignment->scene == NULL);
  assert(assignment->scene_index == -1);
  assert(assignment->Object());
  assert(assignment->Object()->Scene() == this);
  assert(assignment->Label());
  assert(assignment->Label()->Scene() == this);

  // Insert assignment into scene
  assignment->scene = this;
  assignment->scene_index = assignments.NEntries();
  assignments.Insert(assignment);

  // Update object and label
  assignment->Object()->UpdateAfterInsertLabelAssignment(assignment);
  assignment->Label()->UpdateAfterInsertLabelAssignment(assignment);

  // Update assignment
  assignment->UpdateAfterInsert(this);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveLabelAssignment(R3SurfelLabelAssignment *assignment)
{
  // Just checking
  assert(assignment);
  assert(assignment->scene == this);
  assert(assignment->scene_index >= 0);
  assert(assignment->Object());
  assert(assignment->Object()->Scene() == this);
  assert(assignment->Label());
  assert(assignment->Label()->Scene() == this);

  // Update assignment
  assignment->UpdateBeforeRemove(this);

  // Update object and label
  assignment->Object()->UpdateBeforeRemoveLabelAssignment(assignment);
  assignment->Label()->UpdateBeforeRemoveLabelAssignment(assignment);

  // Remove assignment from scene
  assert(assignment->scene == this);
  RNArrayEntry *entry = assignments.KthEntry(assignment->scene_index);
  R3SurfelLabelAssignment *tail = assignments.Tail();
  tail->scene_index = assignment->scene_index;
  assignments.EntryContents(entry) = tail;
  assignments.RemoveTail();
  assignment->scene_index = -1;
  assignment->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertScan(R3SurfelScan *scan)
{
  // Just checking
  assert(scan);

  // Insert label 
  scan->scene = this;
  scan->scene_index = scans.NEntries();
  scans.Insert(scan);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveScan(R3SurfelScan *scan)
{
  // Just checking
  assert(scan);

  // Remove scan from scene
  RNArrayEntry *entry = scans.KthEntry(scan->scene_index);
  R3SurfelScan *tail = scans.Tail();
  tail->scene_index = scan->scene_index;
  scans.EntryContents(entry) = tail;
  scans.RemoveTail();
  scan->scene_index = -1;
  scan->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertImage(R3SurfelImage *image)
{
  // Just checking
  assert(image);

  // Insert label 
  image->scene = this;
  image->scene_index = images.NEntries();
  images.Insert(image);

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveImage(R3SurfelImage *image)
{
  // Just checking
  assert(image);

  // Remove image from scene
  RNArrayEntry *entry = images.KthEntry(image->scene_index);
  R3SurfelImage *tail = images.Tail();
  tail->scene_index = image->scene_index;
  images.EntryContents(entry) = tail;
  images.RemoveTail();
  image->scene_index = -1;
  image->scene = NULL;

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertFeature(R3SurfelFeature *feature)
{
  // Just checking
  assert(feature);

  // Insert label 
  feature->scene = this;
  feature->scene_index = features.NEntries();
  features.Insert(feature);

  // Update all object feature vectors
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    if (object->feature_vector.NValues() > 0) {
      object->feature_vector.Resize(NFeatures());
      feature->UpdateFeatureVector(object, object->feature_vector);
    }
  }

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveFeature(R3SurfelFeature *feature)
{
  // Just checking
  assert(feature);
  assert(feature->scene == this);
  assert(feature->scene_index >= 0);
  assert(feature->scene_index < NFeatures());

  // Remove feature from scene
  RNArrayEntry *entry = features.KthEntry(feature->scene_index);
  R3SurfelFeature *tail = features.Tail();
  tail->scene_index = feature->scene_index;
  features.EntryContents(entry) = tail;
  features.RemoveTail();
  feature->scene_index = -1;
  feature->scene = NULL;

  // Update all object feature vectors
  if (feature != tail) {
    for (int i = 0; i < NObjects(); i++) {
      R3SurfelObject *object = Object(i);
      R3SurfelFeatureVector& vector = object->feature_vector;
      if (vector.NValues() == 0) continue;
      assert(vector.NValues() == NFeatures()+1);
      RNScalar tail_value = vector.Value(vector.NValues()-1);
      object->feature_vector.SetValue(tail->scene_index, tail_value);
      object->feature_vector.Resize(NFeatures());
    }
  }

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertComment(const char *comment)
{
  // Check if already have same comment
  for (int i = 0; i < comments.NEntries(); i++) {
    if (!strcmp(comment, comments[i])) return;
  }
  
  // Insert comment
  comments.Insert(strdup(comment));

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
RemoveComment(const char *comment)
{
  // Remove comment
  for (int i = 0; i < comments.NEntries(); i++) {
    if (!strcmp(comment, comments[i])) {
      free(comments[i]);
      comments.RemoveKth(i);
      break;
    }
  }

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



void R3SurfelScene::
InsertScene(const R3SurfelScene& scene2, 
  R3SurfelObject *parent_object1, 
  R3SurfelLabel *parent_label1,
  R3SurfelNode *parent_node1)
{
  // Get convenient variables
  R3SurfelScene& scene1 = *this;
  R3SurfelTree *tree1 = scene1.Tree();
  R3SurfelTree *tree2 = scene2.Tree();
  R3SurfelDatabase *database1 = tree1->Database();
  R3SurfelDatabase *database2 = tree2->Database();

  // Find parents in scene1
  if (!parent_node1) parent_node1 = tree1->RootNode();
  if (!parent_object1) parent_object1 = scene1.RootObject();
  if (!parent_label1) parent_label1 = scene1.RootLabel();

 // COPY COMMENTS

  // Copy comments from scene2
  for (int i = 0; i < scene2.NComments(); i++) {
    const char *comment2 = scene2.Comment(i);
    scene1.InsertComment(comment2);
  }

 // COPY FEATURES

#if 0
  // Copy label features from scene2
  for (int i = 0; i < scene2.NFeatures(); i++) {
    R3SurfelFeature *feature2 = scene2.Feature(i);
    R3SurfelFeature *feature1 = new R3SurfelFeature(feature2->Name(), feature2->Minimum(), feature2->Maximum());
    scene1.InsertFeature(feature1);
  }
#endif

  // COPY NODES

  // Create surfel tree nodes in scene1
  RNArray<R3SurfelNode *> nodes1;
  nodes1.Insert(tree1->RootNode());
  for (int i = 1; i < tree2->NNodes(); i++) {
    R3SurfelNode *node1 = new R3SurfelNode();
    nodes1.Insert(node1);
  }

  // Copy node stuff from scene2
  for (int i = 1; i < tree2->NNodes(); i++) {
    R3SurfelNode *node1 = nodes1.Kth(i);
    R3SurfelNode *node2 = tree2->Node(i);
    R3SurfelNode *parent2 = node2->Parent();
    R3SurfelNode *parent1 = parent_node1;
    if (parent2 && (!parent2->Name() || strcmp(parent2->Name(), "Root")))
      parent1 = nodes1.Kth(parent2->TreeIndex());
    tree1->InsertNode(node1, parent1);
    node1->SetName(node2->Name());
    for (int j = 0; j < node2->NBlocks(); j++) {
      R3SurfelBlock *block2 = node2->Block(j);
      database2->ReadBlock(block2);
      R3SurfelBlock *block1 = new R3SurfelBlock(*block2);
      database1->InsertBlock(block1);
      node1->InsertBlock(block1);
      database1->ReleaseBlock(block1);
      database2->ReleaseBlock(block2);
    }      
  }

  // COPY OBJECTS

  // Create objects in scene1
  RNArray<R3SurfelObject *> objects1;
  objects1.Insert(scene1.RootObject());
  for (int i = 1; i < scene2.NObjects(); i++) {
    R3SurfelObject *object1 = new R3SurfelObject();
    objects1.Insert(object1);
  }

  // Copy object stuff from scene2
  for (int i = 1; i < scene2.NObjects(); i++) {
    R3SurfelObject *object1 = objects1.Kth(i);
    R3SurfelObject *object2 = scene2.Object(i);
    R3SurfelObject *parent2 = object2->Parent();
    R3SurfelObject *parent1 = parent_object1;
    if (parent2 && (!parent2->Name() || strcmp(parent2->Name(), "Root")))
      parent1 = objects1.Kth(parent2->SceneIndex());
    scene1.InsertObject(object1, parent1);
    object1->SetName(object2->Name());
    object1->SetIdentifier(object2->Identifier());
    object1->SetFeatureVector(object2->feature_vector);
    for (int j = 0; j < object2->NNodes(); j++) {
      R3SurfelNode *node2 = object2->Node(j);
      R3SurfelNode *node1 = nodes1.Kth(node2->TreeIndex());
      object1->InsertNode(node1);
    }
  }

  // COPY LABELS 

  // Create labels in scene1
  RNArray<R3SurfelLabel *> labels1;
  labels1.Insert(scene1.RootLabel());
  for (int i = 1; i < scene2.NLabels(); i++) {
    R3SurfelLabel *label1 = new R3SurfelLabel();
    labels1.Insert(label1);
  }

  // Copy label stuff from scene2
  for (int i = 1; i < scene2.NLabels(); i++) {
    R3SurfelLabel *label1 = labels1.Kth(i);
    R3SurfelLabel *label2 = scene2.Label(i);
    R3SurfelLabel *parent2 = label2->Parent();
    R3SurfelLabel *parent1 = parent_label1;
    if (parent2 && (!parent2->Name() || strcmp(parent2->Name(), "Root")))
      parent1 = labels1.Kth(parent2->SceneIndex());
    label1->SetName(label2->Name());
    label1->SetIdentifier(label2->Identifier());
    label1->SetAssignmentKeystroke(label2->AssignmentKeystroke());
    label1->SetColor(label2->Color());
    label1->SetFlags(label2->Flags());
    scene1.InsertLabel(label1, parent1);
  }

  // COPY OBJECT PROPERTIES

  // Copy object properties from scene2
  for (int i = 0; i < scene2.NObjectProperties(); i++) {
    R3SurfelObjectProperty *property2 = scene2.ObjectProperty(i);
    int type = property2->Type();
    RNArray<R3SurfelObject *> objects1;
    R3SurfelObject *object2 = property2->Object();
    R3SurfelObject *object1 = objects1.Kth(object2->SceneIndex());
    objects1.Insert(object1);
    int noperands1 = 0;
    RNScalar *operands1 = NULL;
    if (property2->NOperands() > 0) {
      noperands1 = property2->NOperands();
      operands1 = new RNScalar [ noperands1 ];
      for (int j = 0; j < noperands1; j++) {
        RNScalar operand2 = property2->Operand(j);
        operands1[j] = operand2;
      }
    }
    R3SurfelObjectProperty *property1 = new R3SurfelObjectProperty(type, object1, operands1, noperands1);
    scene1.InsertObjectProperty(property1);
    delete [] operands1;
  }

  // COPY LABEL PROPERTIES

  // Copy label properties from scene2
  for (int i = 0; i < scene2.NLabelProperties(); i++) {
    R3SurfelLabelProperty *property2 = scene2.LabelProperty(i);
    int type = property2->Type();
    RNArray<R3SurfelLabel *> labels1;
    R3SurfelLabel *label2 = property2->Label();
    R3SurfelLabel *label1 = labels1.Kth(label2->SceneIndex());
    labels1.Insert(label1);
    int noperands1 = 0;
    RNScalar *operands1 = NULL;
    if (property2->NOperands() > 0) {
      noperands1 = property2->NOperands();
      operands1 = new RNScalar [ noperands1 ];
      for (int j = 0; j < noperands1; j++) {
        RNScalar operand2 = property2->Operand(j);
        operands1[j] = operand2;
      }
    }
    R3SurfelLabelProperty *property1 = new R3SurfelLabelProperty(type, label1, operands1, noperands1);
    scene1.InsertLabelProperty(property1);
    delete [] operands1;
  }

  // COPY OBJECT RELATIONSHIPS

  // Copy object relationships from scene2
  for (int i = 0; i < scene2.NObjectRelationships(); i++) {
    R3SurfelObjectRelationship *relationship2 = scene2.ObjectRelationship(i);
    int type = relationship2->Type();
    RNArray<R3SurfelObject *> objects1;
    for (int j = 0; j < relationship2->NObjects(); j++) {
      R3SurfelObject *object2 = relationship2->Object(j);
      R3SurfelObject *object1 = objects1.Kth(object2->SceneIndex());
      objects1.Insert(object1);
    }
    int noperands1 = 0;
    RNScalar *operands1 = NULL;
    if (relationship2->NOperands() > 0) {
      noperands1 = relationship2->NOperands();
      operands1 = new RNScalar [ noperands1 ];
      for (int j = 0; j < noperands1; j++) {
        RNScalar operand2 = relationship2->Operand(j);
        operands1[j] = operand2;
      }
    }
    R3SurfelObjectRelationship *relationship1 = new R3SurfelObjectRelationship(type, objects1, operands1, noperands1);
    scene1.InsertObjectRelationship(relationship1);
    delete [] operands1;
  }

  // COPY LABEL RELATIONSHIPS

  // Copy label relationships from scene2
  for (int i = 0; i < scene2.NLabelRelationships(); i++) {
    R3SurfelLabelRelationship *relationship2 = scene2.LabelRelationship(i);
    int type = relationship2->Type();
    RNArray<R3SurfelLabel *> labels1;
    for (int j = 0; j < relationship2->NLabels(); j++) {
      R3SurfelLabel *label2 = relationship2->Label(j);
      R3SurfelLabel *label1 = labels1.Kth(label2->SceneIndex());
      labels1.Insert(label1);
    }
    int noperands1 = 0;
    RNScalar *operands1 = NULL;
    if (relationship2->NOperands() > 0) {
      noperands1 = relationship2->NOperands();
      operands1 = new RNScalar [ noperands1 ];
      for (int j = 0; j < noperands1; j++) {
        RNScalar operand2 = relationship2->Operand(j);
        operands1[j] = operand2;
      }
    }
    R3SurfelLabelRelationship *relationship1 = new R3SurfelLabelRelationship(type, labels1, operands1);
    scene1.InsertLabelRelationship(relationship1);
    delete [] operands1;
  }

  // COPY ASSIGNMENTS

  // Copy label assignments from scene2
  for (int i = 0; i < scene2.NLabelAssignments(); i++) {
    R3SurfelLabelAssignment *assignment2 = scene2.LabelAssignment(i);
    R3SurfelObject *object2 = assignment2->Object();
    R3SurfelLabel *label2 = assignment2->Label();
    R3SurfelObject *object1 = objects1.Kth(object2->SceneIndex());
    R3SurfelLabel *label1 = labels1.Kth(label2->SceneIndex());
    R3SurfelLabelAssignment *assignment1 = new R3SurfelLabelAssignment(object1, label1, assignment2->Confidence(), assignment2->Originator());
    scene1.InsertLabelAssignment(assignment1);
  }

  // COPY SCANS

  // Copy scans from scene2
  RNArray<R3SurfelScan *> scans1;
  for (int i = 0; i < scene2.NScans(); i++) {
    R3SurfelScan *scan2 = scene2.Scan(i);
    R3SurfelScan *scan1 = new R3SurfelScan(scan2->Name());
    scan1->SetPose(scan2->Pose());
    scan1->SetTimestamp(scan2->Timestamp());
    scan1->SetXFocal(scan2->XFocal());
    scan1->SetYFocal(scan2->YFocal());
    scan1->SetImageDimensions(scan2->ImageWidth(), scan2->ImageHeight());
    scan1->SetImageCenter(scan2->ImageCenter());
    scan1->SetFlags(scan2->Flags());
    R3SurfelNode *node2 = scan2->Node();
    R3SurfelNode *node1 = nodes1.Kth(node2->TreeIndex());
    scan1->SetNode(node1);
    scene1.InsertScan(scan1);
    scans1.Insert(scan1);
  }

  // COPY IMAGES
  
  // Copy images from scene2
  for (int i = 0; i < scene2.NImages(); i++) {
    R3SurfelImage *image2 = scene2.Image(i);
    R3SurfelImage *image1 = new R3SurfelImage(image2->Name());
    const R3CoordSystem *rolling_shutter_poses = image2->RollingShutterPoses();
    const RNScalar *rolling_shutter_timestamps = image2->RollingShutterTimestamps();
    image1->SetRollingShutterPoses(rolling_shutter_poses[0], rolling_shutter_poses[1]);
    image1->SetRollingShutterTimestamps(rolling_shutter_timestamps[0], rolling_shutter_timestamps[1]);
    image1->SetXFocal(image2->XFocal());
    image1->SetYFocal(image2->YFocal());
    image1->SetImageDimensions(image2->ImageWidth(), image2->ImageHeight());
    image1->SetImageCenter(image2->ImageCenter());
    image1->SetDistortionType(image2->DistortionType());
    image1->SetRadialDistortion(image2->RadialDistortion());
    image1->SetTangentialDistortion(image2->TangentialDistortion());
    image1->SetFlags(image2->Flags());
    R3SurfelScan *scan2 = image2->Scan();
    R3SurfelScan *scan1 = (scan2) ? scans1.Kth(scan2->SceneIndex()) : NULL;
    image1->SetScan(scan1);
    scene1.InsertImage(image1);

    // Copy image channels?
    // XXX
  }

  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



///////////////////////////////////////////////////////////////////////
// MEMORY MANAGEMENT FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelScene::
ReadBlocks(void)
{
  // Read blocks for entire tree
  if (!tree) return;
  if (!tree->RootNode()) return;
  tree->RootNode()->ReadBlocks(TRUE);
}

  
  
void R3SurfelScene::
ReleaseBlocks(void)
{
  // Read blocks for entire tree
  if (!tree) return;
  if (!tree->RootNode()) return;
  tree->RootNode()->ReleaseBlocks(TRUE);
}

  

void R3SurfelScene::
CreateMesh(R3Mesh& mesh) const
{
  // Get convenient variables
  R3SurfelTree *tree = Tree();
  if (!tree) return;
  R3SurfelDatabase *database = tree->Database();
  if (!database) return;
  
  // Create disc of triangles for each surfel
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    if (node->NParts() > 0) continue;
    R3SurfelObject *object = (node) ? node->Object(TRUE, TRUE) : NULL;
    while (object && object->Parent() && (object->Parent() != RootObject())) object = object->Parent();
    R3SurfelLabel *label = (object) ? object->CurrentLabel() : NULL;
    int segment_identifier = (object) ? object->SceneIndex() : -1;
    int label_identifier = (label) ? label->Identifier() : -1;
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);

      // Read block
      if (!database->ReadBlock(block)) continue;

      // Create disc of triangles for each surfel
      for (int k = 0; k < block->NSurfels(); k++) {
        int surfel_identifier = block->SurfelIdentifier(k);
        R3Point position = block->SurfelPosition(k);
        RNRgb color = block->SurfelColor(k);
        R3Vector normal = block->SurfelNormal(k);
        R3Vector tangent1 = block->SurfelTangent(k);
        R3Vector tangent2 = normal % tangent1;
        double r1 = block->SurfelRadius(k, 0);
        double r2 = block->SurfelRadius(k, 1);
        if (r1 <= 0) r1 = 0.1;
        if (r2 <= 0) r2 = r1;

        // Create central vertex
        R3MeshVertex *central_vertex = mesh.CreateVertex(position, normal, color, R2Point(0,0));
        if (!central_vertex) continue;
        
        // Create side vertices
        static const int nsides = 6;
        R3MeshVertex *side_vertices[nsides] = { NULL };
        for (int j = 0; j < nsides; j++) {
          double angle = RN_TWO_PI*j/nsides;
          RNScalar c = cos(angle);
          RNScalar s = sin(angle);
          R3Point p = position;
          p += cos(angle) * r1 * tangent1;
          p += sin(angle) * r2 * tangent2;
          R2Point texcoords(c, s);
          side_vertices[j] = mesh.CreateVertex(p, normal, color, texcoords);
        }

        // Create faces
        for (int j = 0; j < nsides; j++) {
          R3MeshFace *face = mesh.CreateFace(central_vertex, side_vertices[j], side_vertices[(j+1)%nsides]);
          mesh.SetFaceMaterial(face, surfel_identifier);
          mesh.SetFaceSegment(face, segment_identifier);
          mesh.SetFaceCategory(face, label_identifier);
        }
      }

      // Release block
      database->ReleaseBlock(block);
    }
  }
}



///////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelScene::
Draw(RNFlags flags) const
{
  // Draw all objects
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    object->Draw(flags);
  } 
}



void R3SurfelScene::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print scene header
  if (prefix) fprintf(fp, "%s", prefix);
  fprintf(fp, "%s", (name) ? name : "Scene");
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");

  // Add indentation to prefix
  char indented_prefix[1024];
  sprintf(indented_prefix, "%s  ", prefix);

  // Print all objects
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    object->Print(fp, indented_prefix, suffix);
  }
}



////////////////////////////////////////////////////////////////////////
// I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelScene::
OpenFile(const char *scene_filename, 
  const char *database_filename, 
  const char *scene_rwaccess,
  const char *database_rwaccess)
{
  // Remember scene file name
  if (this->filename) free(this->filename);
  this->filename = RNStrdup(scene_filename);

  // Remember scene file access mode
  if (this->rwaccess) free(this->rwaccess);
  if (!scene_rwaccess) this->rwaccess = RNStrdup("r");
  else if (!strcmp(scene_rwaccess, "w")) this->rwaccess = RNStrdup("w");
  else if (strstr(scene_rwaccess, "+")) this->rwaccess = RNStrdup("r+");
  else this->rwaccess = RNStrdup("r"); 

  // Open database file
  if (database_filename) {
    if (tree) {
      R3SurfelDatabase *database = tree->Database();
      if (!database->OpenFile(database_filename, database_rwaccess)) {
        return 0;
      }
    }
  }

  // Read scene file
  if (strcmp(this->rwaccess, "w")) {
    if (!ReadFile(scene_filename)) {
      return 0;
    }
  }

  // Return success
  return 1;
}



int R3SurfelScene::
SyncFile(const char *output_scene_filename)
{
  // Check if nothing changed
  if (!flags[R3_SURFEL_SCENE_DIRTY_FLAG]) return 1;

  // Sync surfels
  if (tree) {
    R3SurfelDatabase *database = tree->Database();
    if (database) {
      if (!database->SyncFile()) return 0;
    }
  }

  // Write scene file
  if (output_scene_filename) {
    // Write scene to new file
    if (!WriteFile(output_scene_filename)) {
      return 0;
    }
  }
  else if (this->filename && (strcmp(this->rwaccess, "r"))) {
    // Write scene over original file
    if (!WriteFile(filename)) {
      return 0;
    }
  }

  // Return success
  return 1;
}



int R3SurfelScene::
CloseFile(const char *output_scene_filename)
{
#if 1
  // Clean up
  if (tree) {
    R3SurfelDatabase *database = tree->Database();
    if (database) {
      if (database->PurgeDeletedBlocks()) {
        flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
      }
    }
  }
#endif
  
  // Sync file
  if (!SyncFile(output_scene_filename)) return 0;

  // Close surfel database file
  if (tree) {
    R3SurfelDatabase *database = tree->Database();
    if (database) {
      if (!database->CloseFile()) return 0;
    }
  }

  // Return success
  return 1;
}



int R3SurfelScene::
ReadFile(const char *filename)
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ssa)\n", filename);
    return 0;
  }

  // Read file of appropriate type
  if (!strncmp(extension, ".ssa", 4)) {
    return ReadAsciiFile(filename);
  }
  else if (!strncmp(extension, ".ssx", 4)) {
    return ReadBinaryFile(filename);
  }
  else { 
    RNFail("Unable to read file %s (unrecognized extension: %s)\n", filename, extension); 
    return 0; 
  }

  // Should never get here
  return 0;
}



int R3SurfelScene::
WriteFile(const char *filename) 
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ssa)\n", filename);
    return 0;
  }

  // Write file of appropriate type
  if (!strncmp(extension, ".ssa", 4)) {
    return WriteAsciiFile(filename);
  }
  else if (!strncmp(extension, ".ssx", 4)) {
    return WriteBinaryFile(filename);
  }
  else if (!strncmp(extension, ".arff", 5)) {
    return WriteARFFFile(filename);
  }
  else if (!strncmp(extension, ".tqn", 4)) {
    return WriteTianqiangFile(filename);
  }
  else { 
    RNFail("Unable to write file %s (unrecognized extension: %s)\n", filename, extension); 
    return 0; 
  }

  // Should never get here
  return 0;
}



////////////////////////////////////////////////////////////////////////
// Ascii I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelScene::
ReadAsciiFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "r"))) {
    RNFail("Unable to open file %s\n", filename);
    return 0;
  }

  // Read ascii stream
  int status = ReadAsciiStream(fp);

  // Close file
  fclose(fp);

  // Return status
  return status;
}



int R3SurfelScene::
WriteAsciiFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    RNFail("Unable to open file %s\n", filename);
    return 0;
  }

  // Write ascii stream
  int status = WriteAsciiStream(fp);

  // Close file
  fclose(fp);

  // Return status
  return status;
}



static int 
ReadAsciiString(FILE *fp, char *buffer)
{
  // Read string
  if (fscanf(fp, "%s", buffer) != (unsigned int) 1) {
    RNFail("Unable to read name\n");
    return 0;
  }

  // Restore space characters
  char *bufferp = buffer;
  while (*bufferp) {
    if (*bufferp == '^') *bufferp = '\t';
    if (*bufferp == '&') *bufferp = '\n';
    if (*bufferp == '+') *bufferp = ' ';
    bufferp++;
  }

  // Return success
  return 1;
}



static int 
WriteAsciiString(FILE *fp, const char *name)
{
  // Copy name
  char buffer[1024];
  if (name) strncpy(buffer, name, 1023);
  else strncpy(buffer, "None", 1023);

  // Replace space characters
  char *bufferp = buffer;
  while (*bufferp) {
    if (*bufferp == '\t') *bufferp = '^';
    if (*bufferp == '\n') *bufferp = '&';
    if (*bufferp == ' ') *bufferp = '+';
    bufferp++;
  }

  // Write string
  fprintf(fp, "%s", buffer);

  // Restore space characters
  bufferp = buffer;
  while (*bufferp) {
    if (*bufferp == '^') *bufferp = '\t';
    if (*bufferp == '&') *bufferp = '\n';
    if (*bufferp == '+') *bufferp = ' ';
    bufferp++;
  }

  // Return success
  return 1;
}



int R3SurfelScene::
ReadAsciiStream(FILE *fp)
{
  // Read header
  char buffer[1024], version[1024];
  if (fscanf(fp, "%s%s", buffer, version) != (unsigned int) 2) {
    RNFail("Unable to read scene file %s\n", filename);
    return 0;
  }

  // Check header
  if (strcmp(buffer, "SSA")) {
    RNFail("Wrong header line in Scene file %s\n", filename);
    return 0;
  }

  // Read scene header
  int nnodes;
  int nobjects, nlabels; 
  int nobject_properties, nlabel_properties;
  int nobject_relationships, nlabel_relationships;
  int nassignments, nfeatures, nscans, nimages, ncomments, dummy;
  ReadAsciiString(fp, buffer);
  if (strcmp(buffer, "None")) SetName(buffer);
  fscanf(fp, "%d%d%d%d%d%d%d%d%d%d%d%d", &nnodes, &nobjects, &nlabels, &nfeatures, 
    &nobject_relationships, &nlabel_relationships, &nassignments, &nscans, 
         &nobject_properties, &nlabel_properties, &nimages, &ncomments);
  for (int j = 0; j < 3; j++) fscanf(fp, "%s", buffer);

  // Read transformation (not present in version 1.0)
  if (strcmp(version, "1.0")) {
    double m[16];
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "T")) { RNFail("Error reading xform in %s\n", filename); return 0; }
    for (int i = 0; i < 16; i++) fscanf(fp, "%lf", &m[i]);
    SetTransformation(R3Affine(R4Matrix(m), 0));
  }

  // Read comments
  for (int i = 0; i < ncomments; i++) {
    char comment[1024];
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "C")) { RNFail("Error reading comment %d in %s\n", i, filename); return 0; }
    ReadAsciiString(fp, comment); 
    InsertComment(comment);
  }

  // Create nodes
  RNArray<R3SurfelNode *> read_nodes;
  read_nodes.Insert(tree->RootNode());
  for (int i = 1; i < nnodes; i++) {
    R3SurfelNode *node = new R3SurfelNode();
    read_nodes.Insert(node);
  }

  // Read nodes
  for (int i = 0; i < nnodes; i++) {
    R3SurfelNode *node = read_nodes.Kth(i);
    char node_name[1024];
    int parent_index, nparts, nblocks;
    double complexity, resolution;
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "N")) { RNFail("Error reading node %d in %s\n", i, filename); return 0; }
    ReadAsciiString(fp, node_name); 
    fscanf(fp, "%d%d%d%d%lf%lf", &parent_index, &nparts, &nblocks, &dummy, &complexity, &resolution);
    for (int j = 0; j < 8; j++) fscanf(fp, "%s", buffer);
    if (strcmp(node_name, "None")) node->SetName(node_name);
    node->resolution = resolution;
    for (int j = 0; j < nblocks; j++) {
      int block_index;
      fscanf(fp, "%d", &block_index);
      R3SurfelBlock *block = tree->Database()->Block(block_index);
      node->InsertBlock(block);
    }
    R3SurfelNode *parent = (parent_index >= 0) ? read_nodes.Kth(parent_index) : NULL;
    if (parent) tree->InsertNode(node, parent);
  }

  // Create objects
  RNArray<R3SurfelObject *> read_objects;
  read_objects.Insert(RootObject());
  for (int i = 1; i < nobjects; i++) {
    R3SurfelObject *object = new R3SurfelObject();
    read_objects.Insert(object);
  }

  // Read objects
  for (int i = 0; i < nobjects; i++) {
    R3SurfelObject *object = read_objects.Kth(i);
    char object_name[1024];
    int identifier, parent_index, nparts, nvalues, nnodes;
    double complexity;
    unsigned int flags;
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "O")) { RNFail("Error reading object %d in %s\n", i, filename); return 0; }
    ReadAsciiString(fp, object_name); 
    fscanf(fp, "%d%d%d%d%d%lf%u", &identifier, &parent_index, &nparts, &nnodes, &nvalues, &complexity, &flags);
    for (int j = 0; j < 7; j++) fscanf(fp, "%s", buffer);
    if (strcmp(object_name, "None")) object->SetName(object_name);
    object->SetIdentifier(identifier);
    object->SetFlags(flags);
    R3SurfelFeatureVector vector(nvalues);
    for (int j = 0; j < nvalues; j++) {
      RNScalar value;
      fscanf(fp, "%lf", &value);
      vector.SetValue(j, value);
    }
    object->SetFeatureVector(vector);
    for (int j = 0; j < nnodes; j++) {
      int node_index;
      fscanf(fp, "%d", &node_index);
      R3SurfelNode *node = read_nodes.Kth(node_index);
      object->InsertNode(node);
    }
    R3SurfelObject *parent = (parent_index >= 0) ? read_objects.Kth(parent_index) : NULL;
    if (i > 0) InsertObject(object, parent);
  }

  // Create labels
  RNArray<R3SurfelLabel *> read_labels;
  read_labels.Insert(RootLabel());
  for (int i = 1; i < nlabels; i++) {
    R3SurfelLabel *label = new R3SurfelLabel();
    read_labels.Insert(label);
  }

  // Read labels
  for (int i = 0; i < nlabels; i++) {
    R3SurfelLabel *label = read_labels.Kth(i);
    char label_name[1024];
    int identifier, assignment_key, parent_index, nparts, dummy;
    unsigned int flags;
    double red, green, blue;
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "L")) { RNFail("Error reading label %d in %s\n", i, filename); return 0; }
    ReadAsciiString(fp, label_name); 
    fscanf(fp, "%d%d%d%d%d%lf%lf%lf%u", &identifier, &assignment_key, &dummy, &parent_index, &nparts, &red, &green, &blue, &flags);
    for (int j = 0; j < 3; j++) fscanf(fp, "%s", buffer);
    if (strcmp(label_name, "None")) label->SetName(label_name);
    label->SetIdentifier(identifier);
    label->SetAssignmentKeystroke(assignment_key);
    label->SetColor(RNRgb(red, green, blue));
    label->SetFlags(flags);
    R3SurfelLabel *parent = (parent_index >= 0) ? read_labels.Kth(parent_index) : NULL;
    if (i > 0) InsertLabel(label, parent);
  }

  // Read features
  for (int i = 0; i < nfeatures; i++) {
    char feature_name[1024];
    double minimum, maximum, weight;
    int type, format;
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "F")) { RNFail("Error reading feature %d in %s\n", i, filename); return 0; }
    ReadAsciiString(fp, feature_name); 
    fscanf(fp, "%lf%lf%lf%d%d", &minimum, &maximum, &weight, &type, &format);
    for (int j = 0; j < 1; j++) fscanf(fp, "%s", buffer);
    R3SurfelFeature *feature = NULL;
    if (format == 0) { weight = 1; type = R3_SURFEL_POINTSET_FEATURE_TYPE; }
    if (type == R3_SURFEL_POINTSET_FEATURE_TYPE) {
      feature = new R3SurfelPointSetFeature(feature_name, minimum, maximum, weight);
    }
    else if (type == R3_SURFEL_OVERHEAD_GRID_FEATURE_TYPE) {
      char grid_filename[1024];
      fscanf(fp, "%s", grid_filename);
      feature = new R3SurfelOverheadGridFeature(grid_filename, feature_name, minimum, maximum, weight);
    }
    else {
      feature = new R3SurfelFeature(feature_name, minimum, maximum, weight);
    }
    feature->scene = this;
    feature->scene_index = features.NEntries();
    features.Insert(feature);
  }

  // Read object relationships
  for (int i = 0; i < nobject_relationships; i++) {
    int type, nobjects, noperands;
    RNArray<R3SurfelObject *> objs;
    RNScalar *operands = NULL;
    fscanf(fp, "%s%d%d%d", buffer, &type, &nobjects, &noperands);
    if (strcmp(buffer, "OR")) { RNFail("Error reading object relationship %d in %s\n", i, filename); return 0; }
    for (int j = 0; j < 4; j++) fscanf(fp, "%s", buffer);
    if (nobjects > 0) {
      for (int i = 0; i < nobjects; i++) {
        int object_index;
        fscanf(fp, "%d", &object_index);
        objs.Insert(read_objects.Kth(object_index));
      }
    }
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        fscanf(fp, "%lf", &operands[i]);
      }
    }
    R3SurfelObjectRelationship *relationship = new R3SurfelObjectRelationship(type, objs, operands, noperands);
    InsertObjectRelationship(relationship);
    if (operands) delete [] operands;
  }

  // Read label relationships
  for (int i = 0; i < nlabel_relationships; i++) {
    int type, nlabels, noperands;
    RNArray<R3SurfelLabel *> objs;
    RNScalar *operands = NULL;
    fscanf(fp, "%s%d%d%d", buffer, &type, &nlabels, &noperands);
    if (strcmp(buffer, "LR")) { RNFail("Error reading label relationship %d in %s\n", i, filename); return 0; }
    for (int j = 0; j < 4; j++) fscanf(fp, "%s", buffer);
    if (nlabels > 0) {
      for (int i = 0; i < nlabels; i++) {
        int label_index;
        fscanf(fp, "%d", &label_index);
        objs.Insert(read_labels.Kth(label_index));
      }
    }
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        fscanf(fp, "%lf", &operands[i]);
      }
    }
    R3SurfelLabelRelationship *relationship = new R3SurfelLabelRelationship(type, objs, operands, noperands);
    InsertLabelRelationship(relationship);
    if (operands) delete [] operands;
  }

  // Read assignments
  for (int i = 0; i < nassignments; i++) {
    int indexA, indexB;
    double confidence;
    int originator;
    fscanf(fp, "%s%d%d%lf%d", buffer, &indexA, &indexB, &confidence, &originator);
    if (strcmp(buffer, "A")) { RNFail("Error reading assignment %d in %s\n", i, filename); return 0; }
    for (int j = 0; j < 4; j++) fscanf(fp, "%s", buffer);
    R3SurfelObject *object = read_objects.Kth(indexA);
    R3SurfelLabel *label = read_labels.Kth(indexB);
    R3SurfelLabelAssignment *assignment = new R3SurfelLabelAssignment(object, label, confidence, originator);
    InsertLabelAssignment(assignment);
  }

  // Read scans
  RNArray<R3SurfelScan *> read_scans;
  for (int i = 0; i < nscans; i++) {
    char scan_name[1024];
    unsigned int flags;
    int node_index, width, height;
    double px, py, pz, tx, ty, tz, ux, uy, uz, xfocal, yfocal, xcenter, ycenter, timestamp;
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "S")) { RNFail("Error reading scan %d in %s\n", i, filename); return 0; }
    ReadAsciiString(fp, scan_name); 
    fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%d%d%d%lf%lf%lf%u%lf", &px, &py, &pz, &tx, &ty, &tz, &ux, &uy, &uz, &timestamp, &node_index, &width, &height, &xfocal, &xcenter, &ycenter, &flags, &yfocal);
    for (int j = 0; j < 1; j++) fscanf(fp, "%s", buffer);
    if (xcenter == 0) xcenter = width/2.0;
    if (ycenter == 0) ycenter = height/2.0;
    R3Point viewpoint(px, py, pz);
    R3Vector towards(tx, ty, tz);
    R3Vector up(ux, uy, uz);
    R3CoordSystem pose(viewpoint, R3Triad(towards, up));
    if (yfocal <= 0) yfocal = xfocal;
    R3SurfelScan *scan = new R3SurfelScan();
    scan->SetPose(pose);
    scan->SetXFocal(xfocal);
    scan->SetYFocal(yfocal);
    scan->SetTimestamp(timestamp);
    scan->SetImageDimensions(width, height);
    scan->SetImageCenter(R2Point(xcenter, ycenter));
    scan->SetFlags(flags);
    if (strcmp(scan_name, "None")) scan->SetName(scan_name);
    R3SurfelNode *node = (node_index >= 0)? read_nodes.Kth(node_index) : NULL;
    scan->SetNode(node);
    scan->scene = this;
    scan->scene_index = scans.NEntries();
    scans.Insert(scan);
    read_scans.Insert(scan);
  }

  // Read images
  for (int i = 0; i < nimages; i++) {
    char image_name[1024];
    unsigned int flags;
    int scan_index, width, height, distortion_type, rolling_shutter;
    double xfocal, yfocal, xcenter, ycenter, timestamp;
    double px, py, pz, tx, ty, tz, ux, uy, uz;
    fscanf(fp, "%s", buffer);
    if (strcmp(buffer, "S") && strcmp(buffer, "I")) { RNFail("Error reading image %d in %s\n", i, filename); return 0; }
    ReadAsciiString(fp, image_name); 
    fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%d%d%d%lf%lf%lf%lf%u%d%d", &px, &py, &pz, &tx, &ty, &tz, &ux, &uy, &uz, &timestamp, &scan_index, &width, &height, &xfocal, &yfocal, &xcenter, &ycenter, &flags, &distortion_type, &rolling_shutter);
    for (int j = 0; j < 2; j++) fscanf(fp, "%s", buffer);
    if (xcenter == 0) xcenter = width/2.0;
    if (ycenter == 0) ycenter = height/2.0;
    R3Point viewpoint(px, py, pz);
    R3Vector towards(tx, ty, tz);
    R3Vector up(ux, uy, uz);
    R3CoordSystem pose(viewpoint, R3Triad(towards, up));
    R3SurfelImage *image = new R3SurfelImage();
    image->SetPose(pose);
    image->SetXFocal(xfocal);
    image->SetYFocal(yfocal);
    image->SetTimestamp(timestamp);
    image->SetImageDimensions(width, height);
    image->SetImageCenter(R2Point(xcenter, ycenter));
    image->SetFlags(flags);
    if (strcmp(image_name, "None")) image->SetName(image_name);
    R3SurfelScan *scan = (scan_index >= 0)? read_scans.Kth(scan_index) : NULL;
    image->SetScan(scan);
    image->scene = this;
    image->scene_index = images.NEntries();
    images.Insert(image);

    // Parse extra image parameters
    if (distortion_type != R3_SURFEL_NO_DISTORTION) {
      RNScalar radial_distortion[3] = { 0, 0, 0 };
      RNScalar tangential_distortion[2] = { 0, 0 };
      fscanf(fp, "%lf%lf%lf", &radial_distortion[0], &radial_distortion[1], &radial_distortion[2]);
      fscanf(fp, "%lf%lf", &tangential_distortion[0], &tangential_distortion[1]);
      image->SetDistortionType(distortion_type);
      image->SetRadialDistortion(radial_distortion);
      image->SetTangentialDistortion(tangential_distortion);
    }
    if (rolling_shutter) {
      double px0, py0, pz0, tx0, ty0, tz0, ux0, uy0, uz0;
      double px1, py1, pz1, tx1, ty1, tz1, ux1, uy1, uz1;
      double timestamp0, timestamp1;
      fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &px0, &py0, &pz0, &tx0, &ty0, &tz0, &ux0, &uy0, &uz0);
      fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &px1, &py1, &pz1, &tx1, &ty1, &tz1, &ux1, &uy1, &uz1);
      fscanf(fp, "%lf%lf", &timestamp0, &timestamp1);
      R3Point viewpoint0(px0, py0, pz0);
      R3Vector towards0(tx0, ty0, tz0);
      R3Vector up0(ux0, uy0, uz0);
      R3CoordSystem pose0(viewpoint0, R3Triad(towards0, up0));
      R3Point viewpoint1(px1, py1, pz1);
      R3Vector towards1(tx1, ty1, tz1);
      R3Vector up1(ux1, uy1, uz1);
      R3CoordSystem pose1(viewpoint1, R3Triad(towards1, up1));
      image->SetRollingShutterPoses(pose0, pose1);
      image->SetRollingShutterTimestamps(timestamp0, timestamp1);
    }
  }

  // Read object properties
  for (int i = 0; i < nobject_properties; i++) {
    int type, object_index, noperands;
    RNScalar *operands = NULL;
    fscanf(fp, "%s%d%d%d", buffer, &type, &object_index, &noperands);
    if (strcmp(buffer, "OP")) { RNFail("Error reading object property %d in %s\n", i, filename); return 0; }
    for (int j = 0; j < 4; j++) fscanf(fp, "%s", buffer);
    R3SurfelObject *object = (object_index >= 0) ? read_objects.Kth(object_index) : NULL;
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        fscanf(fp, "%lf", &operands[i]);
      }
    }
    R3SurfelObjectProperty *property = new R3SurfelObjectProperty(type, object, operands, noperands);
    InsertObjectProperty(property);
    if (operands) delete [] operands;
  }

  // Read label properties
  for (int i = 0; i < nlabel_properties; i++) {
    int type, label_index, noperands;
    RNScalar *operands = NULL;
    fscanf(fp, "%s%d%d%d", buffer, &type, &label_index, &noperands);
    if (strcmp(buffer, "LP")) { RNFail("Error reading label property %d in %s\n", i, filename); return 0; }
    for (int j = 0; j < 4; j++) fscanf(fp, "%s", buffer);
    R3SurfelLabel *label = (label_index >= 0) ? read_labels.Kth(label_index) : NULL;
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        fscanf(fp, "%lf", &operands[i]);
      }
    }
    R3SurfelLabelProperty *property = new R3SurfelLabelProperty(type, label, operands, noperands);
    InsertLabelProperty(property);
    if (operands) delete [] operands;
  }

  // Mark scene as clean
  flags.Remove(R3_SURFEL_SCENE_DIRTY_FLAG);

  // Return success
  return 1;
}



int R3SurfelScene::
WriteAsciiStream(FILE *fp)
{
  // Write header
  fprintf(fp, "SSA 1.1\n");

  // Write scene header
  WriteAsciiString(fp, name);
  fprintf(fp, " %d %d %d %d %d %d %d %d %d %d %d %d", 
    tree->NNodes(), NObjects(), NLabels(), NFeatures(),
    NObjectRelationships(), NLabelRelationships(), 
    NLabelAssignments(), NScans(), 
    NObjectProperties(), NLabelProperties(),
    NImages(), NComments());
  for (int j = 0; j < 3; j++) fprintf(fp, " 0");
  fprintf(fp, "\n");

  // Write transformation
  fprintf(fp, "T");
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      fprintf(fp, " %g", Transformation().Matrix()[i][j]);
    }
  }
  fprintf(fp, "\n");

  // Write comments
  for (int i = 0; i < NComments(); i++) {
    fprintf(fp, "C ");
    WriteAsciiString(fp, Comment(i));
    fprintf(fp, "\n");
  }

  // Write nodes
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    int parent_index = (node->Parent()) ? node->Parent()->TreeIndex() : -1;
    fprintf(fp, "N ");
    WriteAsciiString(fp, node->Name());
    fprintf(fp, " %d %d %d %d %g %g", parent_index, node->NParts(), node->NBlocks(), 0, node->Complexity(), node->Resolution());
    for (int j = 0; j < 8; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      fprintf(fp, "%d ", block->DatabaseIndex());
    }
    fprintf(fp, "\n");
  }

  // Write objects
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    int parent_index = (object->Parent()) ? object->Parent()->SceneIndex() : -1;
    // const R3SurfelFeatureVector& feature_vector = object->FeatureVector();
    const R3SurfelFeatureVector& feature_vector = object->feature_vector;
    fprintf(fp, "O ");
    WriteAsciiString(fp, object->Name());
    fprintf(fp, " %d %d %d %d %d %g %u", object->Identifier(), parent_index, object->NParts(), object->NNodes(), 
            feature_vector.NValues(), object->Complexity(), (unsigned int) object->Flags());
    for (int j = 0; j < 7; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
    for (int j = 0; j < feature_vector.NValues(); j++) {
      fprintf(fp, "%g ", feature_vector.Value(j));
    }
    fprintf(fp, "\n");
    for (int j = 0; j < object->NNodes(); j++) {
      R3SurfelNode *node = object->Node(j);
      fprintf(fp, "%d ", node->TreeIndex());
    }
    fprintf(fp, "\n");
  }

  // Write labels
  for (int i = 0; i < NLabels(); i++) {
    R3SurfelLabel *label = Label(i);
    int dummy = 0;
    const RNRgb& color = label->Color();
    int parent_index = (label->Parent()) ? label->Parent()->SceneIndex() : -1;
    fprintf(fp, "L ");
    WriteAsciiString(fp, label->Name());
    fprintf(fp, " %d %d %d %d %d %g %g %g %u", label->Identifier(), label->AssignmentKeystroke(), 
      dummy, parent_index, label->NParts(), color.R(), color.G(), color.B(),
      (unsigned int) label->Flags());
    for (int j = 0; j < 3; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
  }

  // Write features
  for (int i = 0; i < NFeatures(); i++) {
    R3SurfelFeature *feature = Feature(i);
    fprintf(fp, "F ");
    WriteAsciiString(fp, feature->Name());
    fprintf(fp, " %g %g %g %d %d", feature->Minimum(), feature->Maximum(), feature->Weight(), feature->Type(), 1);
    for (int j = 0; j < 1; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
    if (feature->Type() == R3_SURFEL_OVERHEAD_GRID_FEATURE_TYPE) {
      const char *filename = ((R3SurfelOverheadGridFeature *) feature)->filename;
      fprintf(fp, "%s\n", (filename) ? filename : "None");
    }
  }

  // Write object relationships
  for (int i = 0; i < NObjectRelationships(); i++) {
    R3SurfelObjectRelationship *relationship = ObjectRelationship(i);
    fprintf(fp, "OR %d %d %d", relationship->Type(), relationship->NObjects(), relationship->NOperands());
    for (int j = 0; j < 4; j++) fprintf(fp, " 0");
    for (int j = 0; j < relationship->NObjects(); j++) fprintf(fp, " %d", relationship->Object(j)->SceneIndex());
    fprintf(fp, "\n");
    for (int j = 0; j < relationship->NOperands(); j++) fprintf(fp, " %g", relationship->Operand(j));
    fprintf(fp, "\n");
    fprintf(fp, "\n");
  }

  // Write label relationships
  for (int i = 0; i < NLabelRelationships(); i++) {
    R3SurfelLabelRelationship *relationship = LabelRelationship(i);
    fprintf(fp, "LR %d %d %d", relationship->Type(), relationship->NLabels(), relationship->NOperands());
    for (int j = 0; j < 4; j++) fprintf(fp, " 0");
    for (int j = 0; j < relationship->NLabels(); j++) fprintf(fp, " %d", relationship->Label(j)->SceneIndex());
    fprintf(fp, "\n");
    for (int j = 0; j < relationship->NOperands(); j++) fprintf(fp, " %g", relationship->Operand(j));
    fprintf(fp, "\n");
    fprintf(fp, "\n");
  }

  // Write assignments
  for (int i = 0; i < NLabelAssignments(); i++) {
    R3SurfelLabelAssignment *assignment = LabelAssignment(i);
    R3SurfelObject *object = assignment->Object();
    R3SurfelLabel *label = assignment->Label();
    fprintf(fp, "A %d %d %g %d", object->SceneIndex(), label->SceneIndex(), assignment->Confidence(), assignment->Originator());
    for (int j = 0; j < 4; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
  }

  // Write scans
  for (int i = 0; i < NScans(); i++) {
    R3SurfelScan *scan = Scan(i);
    R3Point viewpoint = scan->Viewpoint();
    R3Vector towards = scan->Towards();
    R3Vector up = scan->Up();
    fprintf(fp, "S ");
    WriteAsciiString(fp, scan->Name());
    fprintf(fp, " %g %g %g", viewpoint.X(), viewpoint.Y(), viewpoint.Z());
    fprintf(fp, " %g %g %g", towards.X(), towards.Y(), towards.Z());
    fprintf(fp, " %g %g %g", up.X(), up.Y(), up.Z());
    fprintf(fp, " %.6f %d", scan->Timestamp(), (scan->Node()) ? scan->Node()->TreeIndex() : -1);
    fprintf(fp, " %d %d ", scan->ImageWidth(), scan->ImageHeight());
    fprintf(fp, " %g %g %g ", scan->XFocal(), scan->ImageCenter().X(), scan->ImageCenter().Y());
    fprintf(fp, " %u ", (unsigned int) scan->Flags());
    fprintf(fp, " %g ", scan->YFocal());
    for (int j = 0; j < 1; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
  }

  // Write images
  for (int i = 0; i < NImages(); i++) {
    R3SurfelImage *image = Image(i);
    R3Point viewpoint = image->Viewpoint();
    R3Vector towards = image->Towards();
    R3Vector up = image->Up();
    fprintf(fp, "I ");
    WriteAsciiString(fp, image->Name());
    fprintf(fp, " %g %g %g", viewpoint.X(), viewpoint.Y(), viewpoint.Z());
    fprintf(fp, " %g %g %g", towards.X(), towards.Y(), towards.Z());
    fprintf(fp, " %g %g %g", up.X(), up.Y(), up.Z());
    fprintf(fp, " %.6f", image->Timestamp());
    fprintf(fp, " %d ", (image->Scan()) ? image->Scan()->SceneIndex() : -1);
    fprintf(fp, " %d %d ", image->ImageWidth(), image->ImageHeight());
    fprintf(fp, " %g %g ", image->XFocal(), image->YFocal());
    fprintf(fp, " %g %g ", image->ImageCenter().X(), image->ImageCenter().Y());
    fprintf(fp, " %u ", (unsigned int) image->Flags());
    fprintf(fp, " %d ", image->DistortionType());
    fprintf(fp, " %d ", (image->HasRollingShutter()) ? 1 : 0);
    for (int j = 0; j < 2; j++) fprintf(fp, "0 ");

    // Write extra image parameters
    if (image->DistortionType() != R3_SURFEL_NO_DISTORTION) {
      const RNScalar *radial_distortion = image->RadialDistortion();
      const RNScalar *tangential_distortion = image->TangentialDistortion();
      fprintf(fp, "%g %g %g ", radial_distortion[0], radial_distortion[1], radial_distortion[2]);
      fprintf(fp, "%g %g ", tangential_distortion[0], tangential_distortion[1]);
    }
    if (image->HasRollingShutter()) {
      const R3CoordSystem *rs_poses = image->RollingShutterPoses();
      const RNScalar *rs_timestamps = image->RollingShutterTimestamps();
      R4Matrix camera_to_world0 = rs_poses[0].Matrix();
      R4Matrix camera_to_world1 = rs_poses[1].Matrix();
      R3Point viewpoint0 = camera_to_world0 * R3zero_point;
      R3Point viewpoint1 = camera_to_world1 * R3zero_point;
      R3Vector towards0 = camera_to_world0 * R3negz_vector;
      R3Vector towards1 = camera_to_world1 * R3negz_vector;
      R3Vector up0 = camera_to_world0 * R3posy_vector;
      R3Vector up1 = camera_to_world1 * R3posy_vector;
      fprintf(fp, " %g %g %g ", viewpoint0.X(), viewpoint0.Y(), viewpoint0.Z());
      fprintf(fp, " %g %g %g ", towards0.X(), towards0.Y(), towards0.Z());
      fprintf(fp, " %g %g %g ", up0.X(), up0.Y(), up0.Z());
      fprintf(fp, " %g %g %g ", viewpoint1.X(), viewpoint1.Y(), viewpoint1.Z());
      fprintf(fp, " %g %g %g ", towards1.X(), towards1.Y(), towards1.Z());
      fprintf(fp, " %g %g %g ", up1.X(), up1.Y(), up1.Z());
      fprintf(fp, " %.9f %.9f ", rs_timestamps[0], rs_timestamps[1]);
    }
    fprintf(fp, "\n");
  }

  // Write object properties
  for (int i = 0; i < NObjectProperties(); i++) {
    R3SurfelObjectProperty *property = ObjectProperty(i);
    fprintf(fp, "OP %d %d %d", property->Type(), property->Object()->SceneIndex(), property->NOperands());
    for (int j = 0; j < 4; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
    for (int j = 0; j < property->NOperands(); j++) fprintf(fp, " %g", property->Operand(j));
    fprintf(fp, "\n");
    fprintf(fp, "\n");
  }

  // Write label properties
  for (int i = 0; i < NLabelProperties(); i++) {
    R3SurfelLabelProperty *property = LabelProperty(i);
    fprintf(fp, "LP %d %d %d", property->Type(), property->Label()->SceneIndex(), property->NOperands());
    for (int j = 0; j < 4; j++) fprintf(fp, " 0");
    fprintf(fp, "\n");
    for (int j = 0; j < property->NOperands(); j++) fprintf(fp, " %g", property->Operand(j));
    fprintf(fp, "\n");
    fprintf(fp, "\n");
  }

  // Mark scene as clean
  flags.Remove(R3_SURFEL_SCENE_DIRTY_FLAG);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// BINARY I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelScene::
ReadBinaryFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "r"))) {
    RNFail("Unable to open file %s\n", filename);
    return 0;
  }

  // Read binary stream
  int status = ReadBinaryStream(fp);

  // Close file
  fclose(fp);

  // Return status
  return status;
}



int R3SurfelScene::
WriteBinaryFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    RNFail("Unable to open file %s\n", filename);
    return 0;
  }

  // Write binary stream
  int status = WriteBinaryStream(fp);

  // Close file
  fclose(fp);

  // Return status
  return status;
}



static int 
WriteBinaryString(FILE *fp, const char *name, int size = 256)
{
  // Copy name
  char buffer[1024] = { '\0' };
  if (size > 1024) size = 1024;
  if (name) strncpy(buffer, name, size-1);
  else strncpy(buffer, "None", size-1);
  if (fwrite(buffer, sizeof(char), size, fp) != (unsigned int) size) {
    RNFail("Unable to write name to binary file\n");
    return 0;
  }

  // Return success
  return 1;
}



static int 
WriteBinaryInteger(FILE *fp, int value)
{
  // Write value
  if (fwrite(&value, sizeof(int), 1, fp) != (unsigned int) 1) {
    RNFail("Unable to write integer to binary file\n");
    return 0;
  }

  // Return success
  return 1;
}



static int 
WriteBinaryDouble(FILE *fp, double value)
{
  // Write value
  if (fwrite(&value, sizeof(double), 1, fp) != (unsigned int) 1) {
    RNFail("Unable to write integer to binary file\n");
    return 0;
  }

  // Return success
  return 1;
}



static int 
ReadBinaryString(FILE *fp, char *name, int size = 256)
{
  // Copy name
  if (fread(name, sizeof(char), size, fp) != (unsigned int) size) {
    RNFail("Unable to read name from binary file\n");
    return 0;
  }

  // Return success
  return 1;
}



static int 
ReadBinaryInteger(FILE *fp, int *value)
{
  // Read value
  if (fread(value, sizeof(int), 1, fp) != (unsigned int) 1) {
    RNFail("Unable to read integer from binary file\n");
    return 0;
  }

  // Return success
  return 1;
}



static int 
ReadBinaryDouble(FILE *fp, double *value)
{
  // Read value
  if (fread(value, sizeof(double), 1, fp) != (unsigned int) 1) {
    RNFail("Unable to read integer from binary file\n");
    return 0;
  }

  // Return success
  return 1;
}



int R3SurfelScene::
ReadBinaryStream(FILE *fp) 
{
  // Read file header
  char magic[16];
  if (!ReadBinaryString(fp, magic, 16)) {
    RNFail("Unable to read to %s\n", filename);
    return 0;
  }

  // Check file header
  if (strncmp(magic, "SSB", 3)) {
    RNFail("Invalid header in %s\n", filename);
    return 0;
  }

  // Read scene header
  char name[1024];
  int nnodes;
  int nobjects, nlabels, nfeatures;
  int nobject_properties, nlabel_properties;
  int nobject_relationships, nlabel_relationships;
  int nassignments, nscans, nimages, ncomments, dummy;
  ReadBinaryString(fp, name);
  ReadBinaryInteger(fp, &nnodes);
  ReadBinaryInteger(fp, &nobjects);
  ReadBinaryInteger(fp, &nlabels);
  ReadBinaryInteger(fp, &nfeatures);
  ReadBinaryInteger(fp, &nobject_relationships);
  ReadBinaryInteger(fp, &nlabel_relationships);
  ReadBinaryInteger(fp, &nassignments);
  ReadBinaryInteger(fp, &nscans);
  ReadBinaryInteger(fp, &nobject_properties);
  ReadBinaryInteger(fp, &nlabel_properties);
  ReadBinaryInteger(fp, &nimages);
  ReadBinaryInteger(fp, &ncomments);
  for (int j = 0; j < 3; j++) ReadBinaryInteger(fp, &dummy);

  // Read transformation (not present in version 1.0)
  if (strcmp(magic, "SSB 1.0")) {
    double m[16];
    for (int i = 0; i < 16; i++) ReadBinaryDouble(fp, &m[i]);
    SetTransformation(R3Affine(R4Matrix(m), 0));
  }

  // Read comments
  for (int i = 0; i < ncomments; i++) {
    char comment[1024];
    ReadBinaryString(fp, comment);
    InsertComment(comment);
  }

  // Create nodes
  RNArray<R3SurfelNode *> read_nodes;
  read_nodes.Insert(tree->RootNode());
  for (int i = 1; i < nnodes; i++) {
    R3SurfelNode *node = new R3SurfelNode();
    read_nodes.Insert(node);
  }

  // Read nodes
  for (int i = 0; i < nnodes; i++) {
    R3SurfelNode *node = read_nodes.Kth(i);
    char node_name[1024];
    int parent_index, nparts, nblocks;
    double complexity, resolution;
    ReadBinaryString(fp, node_name); 
    ReadBinaryInteger(fp, &parent_index);
    ReadBinaryInteger(fp, &nparts);
    ReadBinaryInteger(fp, &nblocks);
    ReadBinaryDouble(fp, &complexity);
    ReadBinaryDouble(fp, &resolution);
    for (int j = 0; j < 8; j++) ReadBinaryInteger(fp, &dummy);
    if (strcmp(node_name, "None")) node->SetName(node_name);
    node->resolution = resolution;
    for (int j = 0; j < nblocks; j++) {
      int block_index;
      ReadBinaryInteger(fp, &block_index);
      R3SurfelBlock *block = tree->Database()->Block(block_index);
      node->InsertBlock(block);
    }
    R3SurfelNode *parent = (parent_index >= 0) ? read_nodes.Kth(parent_index) : NULL;
    if (parent) tree->InsertNode(node, parent);
  }

  // Create objects
  RNArray<R3SurfelObject *> read_objects;
  read_objects.Insert(RootObject());
  for (int i = 1; i < nobjects; i++) {
    R3SurfelObject *object = new R3SurfelObject();
    read_objects.Insert(object);
  }

  // Read objects
  for (int i = 0; i < nobjects; i++) {
    R3SurfelObject *object = read_objects.Kth(i);
    char object_name[1024];
    int identifier, parent_index, nparts, nvalues, nnodes, flags;
    double complexity;
    ReadBinaryString(fp, object_name);
    ReadBinaryInteger(fp, &identifier);
    ReadBinaryInteger(fp, &parent_index);
    ReadBinaryInteger(fp, &nparts);
    ReadBinaryInteger(fp, &nnodes);
    ReadBinaryInteger(fp, &nvalues);
    ReadBinaryDouble(fp, &complexity);
    ReadBinaryInteger(fp, &flags);
    for (int j = 0; j < 7; j++) ReadBinaryInteger(fp, &dummy);
    if (strcmp(object_name, "None")) object->SetName(object_name);
    object->SetIdentifier(identifier);
    object->SetFlags(flags);
    R3SurfelFeatureVector vector(nvalues);
    for (int j = 0; j < nvalues; j++) {
      RNScalar value;
      ReadBinaryDouble(fp, &value);
      vector.SetValue(j, value);
    }
    object->SetFeatureVector(vector);
    for (int j = 0; j < nnodes; j++) {
      int node_index;
      ReadBinaryInteger(fp, &node_index);
      R3SurfelNode *node = read_nodes.Kth(node_index);
      object->InsertNode(node);
    }
    R3SurfelObject *parent = (parent_index >= 0) ? read_objects.Kth(parent_index) : NULL;
    if (i > 0) InsertObject(object, parent);
  }

  // Create labels
  RNArray<R3SurfelLabel *> read_labels;
  read_labels.Insert(RootLabel());
  for (int i = 1; i < nlabels; i++) {
    R3SurfelLabel *label = new R3SurfelLabel();
    read_labels.Insert(label);
  }

  // Read labels
  for (int i = 0; i < nlabels; i++) {
    R3SurfelLabel *label = read_labels.Kth(i);
    char label_name[1024];
    int identifier, assignment_key, parent_index, nparts, flags, dummy;
    double red, green, blue;
    ReadBinaryString(fp, label_name);
    ReadBinaryInteger(fp, &identifier);
    ReadBinaryInteger(fp, &assignment_key);
    ReadBinaryInteger(fp, &parent_index);
    ReadBinaryInteger(fp, &nparts);
    ReadBinaryDouble(fp, &red);
    ReadBinaryDouble(fp, &green);
    ReadBinaryDouble(fp, &blue);
    ReadBinaryInteger(fp, &flags);
    for (int j = 0; j < 3; j++) ReadBinaryInteger(fp, &dummy);
    if (strcmp(label_name, "None")) label->SetName(label_name);
    label->SetIdentifier(identifier);
    label->SetAssignmentKeystroke(assignment_key);
    label->SetColor(RNRgb(red, green, blue));
    label->SetFlags(flags);
    R3SurfelLabel *parent = (parent_index >= 0) ? read_labels.Kth(parent_index) : NULL;
    if (i > 0) InsertLabel(label, parent);
  }

  // Read features
  for (int i = 0; i < nfeatures; i++) {
    int type;
    char feature_name[1024];
    double minimum, maximum, weight;
    ReadBinaryString(fp, feature_name);
    ReadBinaryDouble(fp, &minimum);
    ReadBinaryDouble(fp, &maximum);
    ReadBinaryDouble(fp, &weight);
    ReadBinaryInteger(fp, &type);
    for (int j = 0; j < 4; j++) ReadBinaryInteger(fp, &dummy);
    R3SurfelFeature *feature = NULL;
    if (type == R3_SURFEL_POINTSET_FEATURE_TYPE) {
      feature = new R3SurfelPointSetFeature(feature_name, minimum, maximum, weight);
    }
    else if (type == R3_SURFEL_OVERHEAD_GRID_FEATURE_TYPE) {
      char grid_filename[1024];
      ReadBinaryString(fp, grid_filename);
      feature = new R3SurfelOverheadGridFeature(grid_filename, feature_name, minimum, maximum, weight);
    }
    else {
      feature = new R3SurfelFeature(feature_name, minimum, maximum, weight);
    }
    feature->scene = this;
    feature->scene_index = features.NEntries();
    features.Insert(feature);
  }

  // Read object relationships
  for (int i = 0; i < nobject_relationships; i++) {
    int type, nobjects, noperands;
    RNArray<R3SurfelObject *> objs;
    RNScalar *operands = NULL;
    ReadBinaryInteger(fp, &type);
    ReadBinaryInteger(fp, &nobjects);
    ReadBinaryInteger(fp, &noperands);
    for (int j = 0; j < 4; j++) ReadBinaryInteger(fp, &dummy);
    if (nobjects > 0) {
      for (int i = 0; i < nobjects; i++) {
        int object_index;
        ReadBinaryInteger(fp, &object_index);
        objs.Insert(read_objects.Kth(object_index));
      }
    }
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        ReadBinaryDouble(fp, &operands[i]);
      }
    }
    R3SurfelObjectRelationship *relationship = new R3SurfelObjectRelationship(type, objs, operands, noperands);
    InsertObjectRelationship(relationship);
    if (operands) delete [] operands;
  }

  // Read label relationships
  for (int i = 0; i < nlabel_relationships; i++) {
    int type, nlabels, noperands;
    RNArray<R3SurfelLabel *> objs;
    RNScalar *operands = NULL;
    ReadBinaryInteger(fp, &type);
    ReadBinaryInteger(fp, &nlabels);
    ReadBinaryInteger(fp, &noperands);
    for (int j = 0; j < 4; j++) ReadBinaryInteger(fp, &dummy);
    if (nlabels > 0) {
      for (int i = 0; i < nlabels; i++) {
        int label_index;
        ReadBinaryInteger(fp, &label_index);
        objs.Insert(read_labels.Kth(label_index));
      }
    }
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        ReadBinaryDouble(fp, &operands[i]);
      }
    }
    R3SurfelLabelRelationship *relationship = new R3SurfelLabelRelationship(type, objs, operands, noperands);
    InsertLabelRelationship(relationship);
    if (operands) delete [] operands;
  }

  // Read assignments
  for (int i = 0; i < nassignments; i++) {
    int indexA, indexB;
    double confidence;
    int originator;
    ReadBinaryInteger(fp, &indexA);
    ReadBinaryInteger(fp, &indexB);
    ReadBinaryDouble(fp, &confidence);
    ReadBinaryInteger(fp, &originator);
    for (int j = 0; j < 4; j++) ReadBinaryInteger(fp, &dummy);
    R3SurfelObject *object = read_objects.Kth(indexA);
    R3SurfelLabel *label = read_labels.Kth(indexB);
    R3SurfelLabelAssignment *assignment = new R3SurfelLabelAssignment(object, label, confidence, originator);
    InsertLabelAssignment(assignment);
  }

  // Read scans
  RNArray<R3SurfelScan *> read_scans;
  for (int i = 0; i < nscans; i++) {
    char scan_name[1024];
    int node_index, width, height, flags;
    double px, py, pz, tx, ty, tz, ux, uy, uz, xfocal, yfocal, xcenter, ycenter, timestamp;
    ReadBinaryString(fp, scan_name); 
    ReadBinaryDouble(fp, &px);
    ReadBinaryDouble(fp, &py);
    ReadBinaryDouble(fp, &pz);
    ReadBinaryDouble(fp, &tx);
    ReadBinaryDouble(fp, &ty);
    ReadBinaryDouble(fp, &tz);
    ReadBinaryDouble(fp, &ux);
    ReadBinaryDouble(fp, &uy);
    ReadBinaryDouble(fp, &uz);
    ReadBinaryDouble(fp, &timestamp);
    ReadBinaryInteger(fp, &node_index);
    ReadBinaryInteger(fp, &width);
    ReadBinaryInteger(fp, &height);
    ReadBinaryDouble(fp, &xfocal);
    ReadBinaryDouble(fp, &xcenter);
    ReadBinaryDouble(fp, &ycenter);
    ReadBinaryInteger(fp, &flags);
    ReadBinaryDouble(fp, &yfocal);
    for (int j = 0; j < 4; j++) ReadBinaryInteger(fp, &dummy);
    R3SurfelScan *scan = new R3SurfelScan();
    if (strcmp(scan_name, "None")) scan->SetName(scan_name);
    if (xcenter == 0) xcenter = width/2.0;
    if (ycenter == 0) ycenter = height/2.0;
    if (yfocal == 0) yfocal = xfocal;
    scan->SetViewpoint(R3Point(px, py, pz));
    scan->SetOrientation(R3Vector(tx, ty, tz), R3Vector(ux, uy, uz));
    scan->SetTimestamp(timestamp);
    scan->SetXFocal(xfocal);
    scan->SetYFocal(yfocal);
    scan->SetImageDimensions(width, height);
    scan->SetImageCenter(R2Point(xcenter, ycenter));
    scan->SetFlags(flags);
    R3SurfelNode *node = (node_index >= 0) ? read_nodes.Kth(node_index) : NULL;
    scan->SetNode(node);
    InsertScan(scan);
    read_scans.Insert(scan);
  }

  // Read images
  for (int i = 0; i < nimages; i++) {
    char image_name[1024];
    int scan_index, width, height, flags, distortion_type, rolling_shutter;
    double px, py, pz, tx, ty, tz, ux, uy, uz, xfocal, yfocal, xcenter, ycenter, timestamp;
    ReadBinaryString(fp, image_name); 
    ReadBinaryDouble(fp, &px);
    ReadBinaryDouble(fp, &py);
    ReadBinaryDouble(fp, &pz);
    ReadBinaryDouble(fp, &tx);
    ReadBinaryDouble(fp, &ty);
    ReadBinaryDouble(fp, &tz);
    ReadBinaryDouble(fp, &ux);
    ReadBinaryDouble(fp, &uy);
    ReadBinaryDouble(fp, &uz);
    ReadBinaryDouble(fp, &timestamp);
    ReadBinaryInteger(fp, &scan_index);
    ReadBinaryInteger(fp, &width);
    ReadBinaryInteger(fp, &height);
    ReadBinaryDouble(fp, &xfocal);
    ReadBinaryDouble(fp, &yfocal);
    ReadBinaryDouble(fp, &xcenter);
    ReadBinaryDouble(fp, &ycenter);
    ReadBinaryInteger(fp, &flags);
    ReadBinaryInteger(fp, &distortion_type);
    ReadBinaryInteger(fp, &rolling_shutter);
    for (int j = 0; j < 3; j++) ReadBinaryInteger(fp, &dummy);
    R3SurfelImage *image = new R3SurfelImage();
    if (strcmp(image_name, "None")) image->SetName(image_name);
    if (xcenter <= 0) xcenter = width/2.0;
    if (ycenter <= 0) ycenter = height/2.0;
    if (yfocal <= 0) yfocal = xfocal;
    image->SetViewpoint(R3Point(px, py, pz));
    image->SetOrientation(R3Vector(tx, ty, tz), R3Vector(ux, uy, uz));
    image->SetTimestamp(timestamp);
    image->SetXFocal(xfocal);
    image->SetYFocal(yfocal);
    image->SetImageDimensions(width, height);
    image->SetImageCenter(R2Point(xcenter, ycenter));
    image->SetFlags(flags);
    R3SurfelScan *scan = (scan_index >= 0) ? read_scans.Kth(scan_index) : NULL;
    image->SetScan(scan);
    InsertImage(image);

    // Write extra image parameters
    if (distortion_type != R3_SURFEL_NO_DISTORTION) {
      RNScalar radial_distortion[3];
      RNScalar tangential_distortion[2];
      ReadBinaryDouble(fp, &radial_distortion[0]);
      ReadBinaryDouble(fp, &radial_distortion[1]);
      ReadBinaryDouble(fp, &radial_distortion[2]);
      ReadBinaryDouble(fp, &tangential_distortion[0]);
      ReadBinaryDouble(fp, &tangential_distortion[1]);
      image->SetDistortionType(distortion_type);
      image->SetRadialDistortion(radial_distortion);
      image->SetTangentialDistortion(tangential_distortion);
    }
    if (rolling_shutter) {
      R3Point viewpoint0, viewpoint1;
      R3Vector towards0, towards1, up0, up1;
      RNScalar timestamp0, timestamp1;
      ReadBinaryDouble(fp, &viewpoint0[0]);
      ReadBinaryDouble(fp, &viewpoint0[1]);
      ReadBinaryDouble(fp, &viewpoint0[2]);
      ReadBinaryDouble(fp, &towards0[0]);
      ReadBinaryDouble(fp, &towards0[1]);
      ReadBinaryDouble(fp, &towards0[2]);
      ReadBinaryDouble(fp, &up0[0]);
      ReadBinaryDouble(fp, &up0[1]);
      ReadBinaryDouble(fp, &up0[2]);
      ReadBinaryDouble(fp, &viewpoint1[0]);
      ReadBinaryDouble(fp, &viewpoint1[1]);
      ReadBinaryDouble(fp, &viewpoint1[2]);
      ReadBinaryDouble(fp, &towards1[0]);
      ReadBinaryDouble(fp, &towards1[1]);
      ReadBinaryDouble(fp, &towards1[2]);
      ReadBinaryDouble(fp, &up1[0]);
      ReadBinaryDouble(fp, &up1[1]);
      ReadBinaryDouble(fp, &up1[2]);
      ReadBinaryDouble(fp, &timestamp0);
      ReadBinaryDouble(fp, &timestamp1);
      R3CoordSystem pose0(viewpoint0, R3Triad(towards0, up0));
      R3CoordSystem pose1(viewpoint1, R3Triad(towards1, up1));
      image->SetRollingShutterPoses(pose0, pose1);
      image->SetRollingShutterTimestamps(timestamp0, timestamp1);
    }
  }

  // Read object properties
  for (int i = 0; i < nobject_properties; i++) {
    int type, object_index, noperands;
    RNArray<R3SurfelObject *> objs;
    RNScalar *operands = NULL;
    ReadBinaryInteger(fp, &type);
    ReadBinaryInteger(fp, &object_index);
    ReadBinaryInteger(fp, &noperands);
    for (int j = 0; j < 4; j++) ReadBinaryInteger(fp, &dummy);
    R3SurfelObject *object = (object_index >= 0) ? read_objects.Kth(object_index) : NULL;
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        ReadBinaryDouble(fp, &operands[i]);
      }
    }
    R3SurfelObjectProperty *property = new R3SurfelObjectProperty(type, object, operands, noperands);
    InsertObjectProperty(property);
    if (operands) delete [] operands;
  }

  // Read label properties
  for (int i = 0; i < nlabel_properties; i++) {
    int type, label_index, noperands;
    RNArray<R3SurfelLabel *> objs;
    RNScalar *operands = NULL;
    ReadBinaryInteger(fp, &type);
    ReadBinaryInteger(fp, &label_index);
    ReadBinaryInteger(fp, &noperands);
    for (int j = 0; j < 4; j++) ReadBinaryInteger(fp, &dummy);
    R3SurfelLabel *label = (label_index >= 0) ? read_labels.Kth(label_index) : NULL;
    if (noperands > 0) {
      operands = new RNScalar [ noperands ];
      for (int i = 0; i < noperands; i++) {
        ReadBinaryDouble(fp, &operands[i]);
      }
    }
    R3SurfelLabelProperty *property = new R3SurfelLabelProperty(type, label, operands, noperands);
    InsertLabelProperty(property);
    if (operands) delete [] operands;
  }

  // Mark scene as clean
  flags.Remove(R3_SURFEL_SCENE_DIRTY_FLAG);

  // Return success
  return 1;
}



int R3SurfelScene::
WriteBinaryStream(FILE *fp) 
{
  // Write file header
  if (!WriteBinaryString(fp, "SSB 1.1", 16)) {
    RNFail("Unable to write to %s\n", filename);
    return 0;
  }

  // Write scene header
  WriteBinaryString(fp, name);
  WriteBinaryInteger(fp, tree->NNodes());
  WriteBinaryInteger(fp, NObjects());
  WriteBinaryInteger(fp, NLabels());
  WriteBinaryInteger(fp, NFeatures());
  WriteBinaryInteger(fp, NObjectRelationships());
  WriteBinaryInteger(fp, NLabelRelationships());
  WriteBinaryInteger(fp, NLabelAssignments());
  WriteBinaryInteger(fp, NScans());
  WriteBinaryInteger(fp, NObjectProperties());
  WriteBinaryInteger(fp, NLabelProperties());
  WriteBinaryInteger(fp, NImages());
  WriteBinaryInteger(fp, NComments());
  for (int j = 0; j < 3; j++) WriteBinaryInteger(fp, 0);

  // Write transformation
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      WriteBinaryDouble(fp, Transformation().Matrix()[i][j]);
    }
  }

  // Write comments
  for (int i = 0; i < NComments(); i++) {
    WriteBinaryString(fp, Comment(i));
  }

  // Write nodes
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    int parent_index = (node->Parent()) ? node->Parent()->TreeIndex() : -1;
    WriteBinaryString(fp, node->Name());
    WriteBinaryInteger(fp, parent_index);
    WriteBinaryInteger(fp, node->NParts());
    WriteBinaryInteger(fp, node->NBlocks());
    WriteBinaryDouble(fp, node->Complexity());
    WriteBinaryDouble(fp, node->Resolution());
    for (int j = 0; j < 8; j++) WriteBinaryInteger(fp, 0);
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      WriteBinaryInteger(fp, block->DatabaseIndex());
    }
  }

  // Write objects
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    int parent_index = (object->Parent()) ? object->Parent()->SceneIndex() : -1;
    const R3SurfelFeatureVector& feature_vector = object->FeatureVector();
    WriteBinaryString(fp, object->Name());
    WriteBinaryInteger(fp, object->Identifier());
    WriteBinaryInteger(fp, parent_index);
    WriteBinaryInteger(fp, object->NParts());
    WriteBinaryInteger(fp, object->NNodes());
    WriteBinaryInteger(fp, feature_vector.NValues());
    WriteBinaryDouble(fp, object->Complexity());
    WriteBinaryInteger(fp, object->Flags());
    for (int j = 0; j < 7; j++) WriteBinaryInteger(fp, 0);
    for (int j = 0; j < feature_vector.NValues(); j++) {
      WriteBinaryDouble(fp, feature_vector.Value(j));
    }
    for (int j = 0; j < object->NNodes(); j++) {
      R3SurfelNode *node = object->Node(j);
      WriteBinaryInteger(fp, node->TreeIndex());
    }
  }

  // Write labels
  for (int i = 0; i < NLabels(); i++) {
    R3SurfelLabel *label = Label(i);
    const RNRgb& color = label->Color();
    int parent_index = (label->Parent()) ? label->Parent()->SceneIndex() : -1;
    WriteBinaryString(fp, label->Name());
    WriteBinaryInteger(fp, label->Identifier());
    WriteBinaryInteger(fp, label->AssignmentKeystroke());
    WriteBinaryInteger(fp, parent_index);
    WriteBinaryInteger(fp, label->NParts());
    WriteBinaryDouble(fp, color.R());
    WriteBinaryDouble(fp, color.G());
    WriteBinaryDouble(fp, color.B());
    WriteBinaryInteger(fp, label->Flags());
    for (int j = 0; j < 3; j++) WriteBinaryInteger(fp, 0);
  }

  // Write features
  for (int i = 0; i < NFeatures(); i++) {
    R3SurfelFeature *feature = Feature(i);
    WriteBinaryString(fp, feature->Name());
    WriteBinaryDouble(fp, feature->Minimum());
    WriteBinaryDouble(fp, feature->Maximum());
    WriteBinaryDouble(fp, feature->Weight());
    WriteBinaryInteger(fp, feature->Type());
    for (int j = 0; j < 4; j++) WriteBinaryInteger(fp, 0);
    if (feature->Type() == R3_SURFEL_OVERHEAD_GRID_FEATURE_TYPE) {
      WriteBinaryString(fp, ((R3SurfelOverheadGridFeature *) feature)->filename);
    }
  }

  // Write object relationships
  for (int i = 0; i < NObjectRelationships(); i++) {
    R3SurfelObjectRelationship *relationship = ObjectRelationship(i);
    WriteBinaryInteger(fp, relationship->Type());
    WriteBinaryInteger(fp, relationship->NObjects());
    WriteBinaryInteger(fp, relationship->NOperands());
    for (int j = 0; j < 4; j++) WriteBinaryInteger(fp, 0);
    for (int j = 0; j < relationship->NObjects(); j++) 
      WriteBinaryInteger(fp, relationship->Object(j)->SceneIndex());
    for (int j = 0; j < relationship->NOperands(); j++) 
      WriteBinaryDouble(fp, relationship->Operand(j));
  }

  // Write label relationships
  for (int i = 0; i < NLabelRelationships(); i++) {
    R3SurfelLabelRelationship *relationship = LabelRelationship(i);
    WriteBinaryInteger(fp, relationship->Type());
    WriteBinaryInteger(fp, relationship->NLabels());
    WriteBinaryInteger(fp, relationship->NOperands());
    for (int j = 0; j < 4; j++) WriteBinaryInteger(fp, 0);
    for (int j = 0; j < relationship->NLabels(); j++) 
      WriteBinaryInteger(fp, relationship->Label(j)->SceneIndex());
    for (int j = 0; j < relationship->NOperands(); j++) 
      WriteBinaryDouble(fp, relationship->Operand(j));
  }

  // Write assignments
  for (int i = 0; i < NLabelAssignments(); i++) {
    R3SurfelLabelAssignment *assignment = LabelAssignment(i);
    R3SurfelObject *object = assignment->Object();
    R3SurfelLabel *label = assignment->Label();
    WriteBinaryInteger(fp, object->SceneIndex());
    WriteBinaryInteger(fp, label->SceneIndex());
    WriteBinaryDouble(fp, assignment->Confidence());
    WriteBinaryInteger(fp, assignment->Originator());
    for (int j = 0; j < 4; j++) WriteBinaryInteger(fp, 0);
  }

  // Write scans
  for (int i = 0; i < NScans(); i++) {
    R3SurfelScan *scan = Scan(i);
    WriteBinaryString(fp, scan->Name());
    WriteBinaryDouble(fp, scan->Viewpoint().X());
    WriteBinaryDouble(fp, scan->Viewpoint().Y());
    WriteBinaryDouble(fp, scan->Viewpoint().Z());
    WriteBinaryDouble(fp, scan->Towards().X());
    WriteBinaryDouble(fp, scan->Towards().Y());
    WriteBinaryDouble(fp, scan->Towards().Z());
    WriteBinaryDouble(fp, scan->Up().X());
    WriteBinaryDouble(fp, scan->Up().Y());
    WriteBinaryDouble(fp, scan->Up().Z());
    WriteBinaryDouble(fp, scan->Timestamp());
    WriteBinaryInteger(fp, (scan->Node()) ? scan->Node()->TreeIndex() : -1);
    WriteBinaryInteger(fp, scan->ImageWidth());
    WriteBinaryInteger(fp, scan->ImageHeight());
    WriteBinaryDouble(fp, scan->XFocal());
    WriteBinaryDouble(fp, scan->ImageCenter().X());
    WriteBinaryDouble(fp, scan->ImageCenter().Y());
    WriteBinaryInteger(fp, scan->Flags());
    WriteBinaryDouble(fp, scan->YFocal());
    for (int j = 0; j < 4; j++) WriteBinaryInteger(fp, 0);
  }

  // Write images
  for (int i = 0; i < NImages(); i++) {
    R3SurfelImage *image = Image(i);
    WriteBinaryString(fp, image->Name());
    WriteBinaryDouble(fp, image->Viewpoint().X());
    WriteBinaryDouble(fp, image->Viewpoint().Y());
    WriteBinaryDouble(fp, image->Viewpoint().Z());
    WriteBinaryDouble(fp, image->Towards().X());
    WriteBinaryDouble(fp, image->Towards().Y());
    WriteBinaryDouble(fp, image->Towards().Z());
    WriteBinaryDouble(fp, image->Up().X());
    WriteBinaryDouble(fp, image->Up().Y());
    WriteBinaryDouble(fp, image->Up().Z());
    WriteBinaryDouble(fp, image->Timestamp());
    WriteBinaryInteger(fp, (image->scan) ? image->scan->SceneIndex() : -1);
    WriteBinaryInteger(fp, image->ImageWidth());
    WriteBinaryInteger(fp, image->ImageHeight());
    WriteBinaryDouble(fp, image->XFocal());
    WriteBinaryDouble(fp, image->YFocal());
    WriteBinaryDouble(fp, image->ImageCenter().X());
    WriteBinaryDouble(fp, image->ImageCenter().Y());
    WriteBinaryInteger(fp, image->Flags());
    WriteBinaryInteger(fp, image->DistortionType());
    WriteBinaryInteger(fp, (image->HasRollingShutter()) ? 1 : 0);
    for (int j = 0; j < 3; j++) WriteBinaryInteger(fp, 0);

    // Write extra image parameters
    if (image->DistortionType() != R3_SURFEL_NO_DISTORTION) {
      const RNScalar *radial_distortion = image->RadialDistortion();
      const RNScalar *tangential_distortion = image->TangentialDistortion();
      WriteBinaryDouble(fp, radial_distortion[0]);
      WriteBinaryDouble(fp, radial_distortion[1]);
      WriteBinaryDouble(fp, radial_distortion[2]);
      WriteBinaryDouble(fp, tangential_distortion[0]);
      WriteBinaryDouble(fp, tangential_distortion[1]);
    }
    if (image->HasRollingShutter()) {
      const R3CoordSystem *rs_poses = image->RollingShutterPoses();
      const RNScalar *rs_timestamps = image->RollingShutterTimestamps();
      R4Matrix camera_to_world0 = rs_poses[0].Matrix();
      R4Matrix camera_to_world1 = rs_poses[1].Matrix();
      R3Point viewpoint0 = camera_to_world0 * R3zero_point;
      R3Point viewpoint1 = camera_to_world1 * R3zero_point;
      R3Vector towards0 = camera_to_world0 * R3negz_vector;
      R3Vector towards1 = camera_to_world1 * R3negz_vector;
      R3Vector up0 = camera_to_world0 * R3posy_vector;
      R3Vector up1 = camera_to_world1 * R3posy_vector;
      WriteBinaryDouble(fp, viewpoint0.X());
      WriteBinaryDouble(fp, viewpoint0.Y());
      WriteBinaryDouble(fp, viewpoint0.Z());
      WriteBinaryDouble(fp, towards0.X());
      WriteBinaryDouble(fp, towards0.Y());
      WriteBinaryDouble(fp, towards0.Z());
      WriteBinaryDouble(fp, up0.X());
      WriteBinaryDouble(fp, up0.Y());
      WriteBinaryDouble(fp, up0.Z());
      WriteBinaryDouble(fp, viewpoint1.X());
      WriteBinaryDouble(fp, viewpoint1.Y());
      WriteBinaryDouble(fp, viewpoint1.Z());
      WriteBinaryDouble(fp, towards1.X());
      WriteBinaryDouble(fp, towards1.Y());
      WriteBinaryDouble(fp, towards1.Z());
      WriteBinaryDouble(fp, up1.X());
      WriteBinaryDouble(fp, up1.Y());
      WriteBinaryDouble(fp, up1.Z());
      WriteBinaryDouble(fp, rs_timestamps[0]);
      WriteBinaryDouble(fp, rs_timestamps[1]);
    }
  }

  // Write object properties
  for (int i = 0; i < NObjectProperties(); i++) {
    R3SurfelObjectProperty *property = ObjectProperty(i);
    WriteBinaryInteger(fp, property->Type());
    WriteBinaryInteger(fp, property->Object()->SceneIndex());
    WriteBinaryInteger(fp, property->NOperands());
    for (int j = 0; j < 4; j++) WriteBinaryInteger(fp, 0);
    for (int j = 0; j < property->NOperands(); j++) 
      WriteBinaryDouble(fp, property->Operand(j));
  }

  // Write label properties
  for (int i = 0; i < NLabelProperties(); i++) {
    R3SurfelLabelProperty *property = LabelProperty(i);
    WriteBinaryInteger(fp, property->Type());
    WriteBinaryInteger(fp, property->Label()->SceneIndex());
    WriteBinaryInteger(fp, property->NOperands());
    for (int j = 0; j < 4; j++) WriteBinaryInteger(fp, 0);
    for (int j = 0; j < property->NOperands(); j++) 
      WriteBinaryDouble(fp, property->Operand(j));
  }

  // Mark scene as clean
  flags.Remove(R3_SURFEL_SCENE_DIRTY_FLAG);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// ARFF I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelScene::
WriteARFFFile(const char *filename) 
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    RNFail("Unable to open file %s\n", filename);
    return 0;
  }

  // Write header
  fprintf(fp, "@relation UGOR\n");

  // Write features
  for (int i = 0; i < NFeatures(); i++) {
    R3SurfelFeature *feature = Feature(i);
    fprintf(fp, "@attribute %s real\n", feature->Name());
  }

  // Write labels
  fprintf(fp, "@attribute Label { ");
  for (int i = 0; i < NLabels(); i++) {
    R3SurfelLabel *label = Label(i);
    if (i != 0) fprintf(fp, ", ");
    fprintf(fp, "%s ", label->Name());
  }
  fprintf(fp, "}\n");

  // Write data
  fprintf(fp, "@data\n");
  for (int i = 1; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    R3SurfelLabel *label = object->GroundTruthLabel();
    if (!label) label = object->HumanLabel();
    const R3SurfelFeatureVector& vector = object->FeatureVector();
    const char *name = (object->Name()) ? object->Name() : "None";
    const R3Point& centroid = object->Centroid();
    fprintf(fp, "%% %s %g %g %g\n", name, centroid.X(), centroid.Y(), centroid.Z());
    for (int j = 0; j < vector.NValues(); j++) 
      fprintf(fp, "%12.6f ", vector.Value(j));
    fprintf(fp, "%s\n", (label) ? label->Name() : "Unknown");;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// TIANQIANG I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

static int 
WriteOffFile(R3SurfelObject *object, const char *filename)
{
  // Get pointset
  R3SurfelPointSet *pointset = object->PointSet();
  if (!pointset) return 0;

  // Create mesh
  R3Mesh mesh;
  for (int i = 0; i < pointset->NPoints(); i++) {
    R3SurfelPoint *point = pointset->Point(i);
    R3Point position = point->Position();
    R3Point position1 = position; position1[0] -= 1E-3;
    R3Point position2 = position; position2[0] += 1E-3;
    R3Point position3 = position; position3[1] += 1E-3;
    R3MeshVertex *v1 = mesh.CreateVertex(position1);
    R3MeshVertex *v2 = mesh.CreateVertex(position2);
    R3MeshVertex *v3 = mesh.CreateVertex(position3);
    mesh.CreateFace(v1, v2, v3);
  }

  // Delete pointset
  delete pointset;

  // Write mesh
  if (!mesh.WriteFile(filename)) return 0;

  // Return succes
  return 1;
}



int R3SurfelScene::
WriteTianqiangFile(const char *filename) 
{
  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    RNFail("Unable to open file %s\n", filename);
    return 0;
  }
  
  // Get scene name
  char scene_name_buffer[2048];
  strncpy(scene_name_buffer, filename, 2047);
  char *scene_name = strrchr(scene_name_buffer, '.');
  if (scene_name) *scene_name = '\0';
  scene_name = strrchr(scene_name_buffer, '/');
  if (scene_name) scene_name++;
  else scene_name = scene_name_buffer;

  // Allocate node and leaf index
  int node_count = 0;
  int leaf_count = 0;
  int *node_index = new int [ NObjects() ];
  int *leaf_index = new int [ NObjects() ];
  for (int i = 0; i < NObjects(); i++) node_index[i] = -1;
  for (int i = 0; i < NObjects(); i++) leaf_index[i] = -1;

  // Compute node and leaf index
  RNArray<R3SurfelObject *> index_stack;
  index_stack.Insert(RootObject());
  while (!index_stack.IsEmpty()) {
    // Pop object off stack
    R3SurfelObject *object = index_stack.Tail();
    index_stack.RemoveTail();

    // Push children on stack
    for (int j = 0; j < object->NParts(); j++) {
      index_stack.Insert(object->Part(j));
    }

    // Check object
    if (object->NParts() == 0) {
      if (object->Parent() == RootObject()) {
        if (!object->GroundTruthLabel() && !object->HumanLabel()) {
          continue;
        }
      }
    }

    // Add object to node index
    node_index[object->SceneIndex()] = node_count++;

    // Add object to leaf index
    if (object->NParts() == 0) {
      leaf_index[object->SceneIndex()] = leaf_count++;
    }
  }

  // Create off directory
  char mkdir_command[4096];
  sprintf(mkdir_command, "mkdir -p %s_off_files", scene_name);
  system(mkdir_command);

  // Write off files
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    if (leaf_index[i] < 0) continue;
    char off_filename[4096];
    sprintf(off_filename, "%s_off_files/%d.off", scene_name, leaf_index[i]);
    if (!WriteOffFile(object, off_filename)) return 0;
  }

  // Write header
  fprintf(fp, "root 0\n");
  fprintf(fp, "scene_name %s\n", scene_name);
  
  // Write objects
  RNArray<R3SurfelObject *> object_stack;
  object_stack.Insert(RootObject());
  while (!object_stack.IsEmpty()) {
    // Pop object off stack
    R3SurfelObject *object = object_stack.Tail();
    object_stack.RemoveTail();

    // Push children on stack
    for (int j = 0; j < object->NParts(); j++) {
      object_stack.Insert(object->Part(j));
    }

    // Check object
    if (node_index[object->SceneIndex()] < 0) continue;

    // Write object header
    fprintf(fp, "newModel %d\n", node_index[object->SceneIndex()]);

    // Write parent 
    R3SurfelObject *parent = object->Parent();
    int parent_index = (parent) ? node_index[parent->SceneIndex()] : -1;
    fprintf(fp, "parent %d\n", parent_index);

    // Write children
    fprintf(fp, "children ");
    for (int j = 0; j < object->NParts(); j++) {
      R3SurfelObject *part = object->Part(j);
      if (node_index[part->SceneIndex()] < 0) continue;
      fprintf(fp, "%d ", node_index[part->SceneIndex()]);
    }
    fprintf(fp, "\n");

    // Write leaf group
    fprintf(fp, "leaf_group");
    RNArray<R3SurfelObject *> part_stack;
    part_stack.Insert(object);
    while (!part_stack.IsEmpty()) {
      R3SurfelObject *part = part_stack.Tail();
      part_stack.RemoveTail();
      for (int j = 0; j < part->NParts(); j++) part_stack.Insert(part->Part(j));
      int g = leaf_index[part->SceneIndex()];
      if (g != -1) fprintf(fp, " %d", g);
    }
    fprintf(fp, "\n");
    
    // Write label
    R3SurfelLabel *label = object->GroundTruthLabel();
    if (!label) label = object->HumanLabel();
    if (!label && (object == RootObject())) label = RootLabel();
    if (label) fprintf(fp, "label %d %s\n", label->SceneIndex(), label->Name());;

    // Write feature vector
    // const R3SurfelFeatureVector& vector = object->FeatureVector();
    // fprintf(fp, "feature_vector ");
    // for (int j = 0; j < vector.NValues(); j++) 
    //   fprintf(fp, "%12.6f ", vector.Value(j));
    // fprintf(fp, "\n");
  }
  
  // Delete temporary data
  delete [] node_index;
  delete [] leaf_index;

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



} // namespace gaps
