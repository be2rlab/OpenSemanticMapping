////////////////////////////////////////////////////////////////////////
// Source file for code that does segmentation of points
////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Utils/R3Utils.h"



//////////////////////////////////////////////////////////////////////
// Namespace
////////////////////////////////////////////////////////////////////////

namespace gaps {

  

//////////////////////////////////////////////////////////////////////
// Parameters
////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////
// R3SegmentationPoint member functions
////////////////////////////////////////////////////////////////////////

R3SegmentationPoint::
R3SegmentationPoint(void)
  : depth(0),
    position(0,0,0),
    normal(0,0,0),
    tangent(0,0,0),
    radius1(0),
    radius2(0),
    elevation(0),
    timestamp(0),
    area(0),
    color(0,0,0),
    category_identifier(-1),
    category_confidence(0),
    attribute(0),
    boundary(0),
    identifier(0),
    neighbors(),
    cluster(NULL),
    cluster_affinity(0),
    cluster_index(-1),
    data_index(-1),
    mark(0)
{
}



////////////////////////////////////////////////////////////////////////
// R3SegmentationPrimitive member functions
////////////////////////////////////////////////////////////////////////

R3SegmentationPrimitive::
R3SegmentationPrimitive(int primitive_type)
  : primitive_type(primitive_type),
    bbox(R3null_box),
    centroid(R3zero_point),
    line(R3null_line),
    plane(R3null_plane)
{
}



R3SegmentationPrimitive::
R3SegmentationPrimitive(const R3SegmentationPrimitive& primitive)
  : primitive_type(primitive.primitive_type),
    bbox(primitive.bbox),
    centroid(primitive.centroid),
    line(primitive.line),
    plane(primitive.plane)
{
}



R3SegmentationPrimitive::
R3SegmentationPrimitive(R3SegmentationPoint *seed_point, const RNArray<R3SegmentationPoint *> *points)
  : primitive_type(R3_SEGMENTATION_NULL_PRIMITIVE_TYPE),
    bbox(R3null_box),
    centroid(R3zero_point),
    line(R3null_line),
    plane(R3null_plane)
{
  // Initialize primitive based on points
  Update(seed_point, points); 
}



RNLength R3SegmentationPrimitive::
Distance(const R3Point& position) const
{
  // Return distance from primitive to point
  if (primitive_type == R3_SEGMENTATION_POINT_PRIMITIVE_TYPE) return R3Distance(centroid, position);
  else if (primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) return R3Distance(line, position);
  else if (primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) return R3Distance(plane, position);
#if 0
  else if (primitive_type == R3_SEGMENTATION_PLANAR_GRID_PRIMITIVE_TYPE) {
    R2R3SegmentationPoint grid_position = planar_grid.GridPosition(position);
    int ix = (int) (grid_position.X() + 0.5);
    if ((ix < 0) || (ix >= planar_grid.XResolution())) return RN_INFINITY;
    int iy = (int) (grid_position.Y() + 0.5);
    if ((iy < 0) || (iy >= planar_grid.YResolution())) return RN_INFINITY;
    RNScalar value = planar_grid.GridValue(ix, iy);
    if (value <= 0) return RN_INFINITY;
    return R3Distance(planar_grid.Plane(), position);
  }
#endif
  else {
    RNAbort("Unrecognized primitive type");
    return RN_INFINITY;
  }
}



void R3SegmentationPrimitive::
Update(const R3Point& point)
{
  // Set everything
  primitive_type = R3_SEGMENTATION_POINT_PRIMITIVE_TYPE;
  this->centroid = point;
  line = R3null_line;
  plane = R3null_plane;
}



void R3SegmentationPrimitive::
Update(const R3Line& line)
{
  // Set everything
  primitive_type = R3_SEGMENTATION_LINE_PRIMITIVE_TYPE;
  centroid = R3zero_point;
  centroid.Project(line);
  this->line = line;
  plane = R3null_plane;
}



void R3SegmentationPrimitive::
Update(const R3Plane& plane)
{
  // Set everything
  primitive_type = R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE;
  centroid = R3zero_point;
  centroid.Project(plane);
  line = R3null_line;
  this->plane = plane;
}



void R3SegmentationPrimitive::
Update(R3SegmentationPoint *seed_point, const RNArray<R3SegmentationPoint *> *points)
{
  // Remember stuff about primitive (so can set same orientation)
  R3Vector previous_vector(0,0,0);
  if (primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) previous_vector = line.Vector();
  if (primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) previous_vector = plane.Normal();

  // Update bounding box
  bbox = R3null_box;
  if (points) {
    for (int i = 0; i < points->NEntries(); i++) {
      R3SegmentationPoint *point = points->Kth(i);
      bbox.Union(point->position);
    }
  }

  // Initialize everything
  if (seed_point) {
    R3Point seed_position = seed_point->position;
    centroid = seed_position;
    line.Reset(seed_position, line.Vector()); 
    plane.Reset(seed_position, seed_point->normal);
    bbox.Union(seed_position);
  }
  else {
    // Temporary
    RNAbort("Need seed point");
  }

  // Update based on points
  if (points && (points->NEntries() > 0)) {
    // Allocate arrays of point positions and weights
    const int max_positions = 1024;
    R3Point *positions = new R3Point [ max_positions ];
    RNScalar *weights = new RNScalar [ max_positions ];

    // Fill arrays of point positions and weights
    int npositions = 0;
    int skip = points->NEntries() / max_positions + 1;
    for (int i = 0; i < points->NEntries(); i += skip) {
      R3SegmentationPoint *point = points->Kth(i);
      if (npositions >= max_positions-1) break;
      positions[npositions] = point->position;
      if (primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE)
        weights[npositions] = fabs(plane.Normal().Dot(point->normal));
      else if (primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE)
        weights[npositions] = 1.0 - fabs(plane.Normal().Dot(point->normal));
      else weights[npositions] = 1.0;
      npositions++;
    }

    // Add seed point with 20% of the total weight
    if (seed_point) {
      positions[npositions] = seed_point->position;
      weights[npositions] = 0.2 * points->NEntries();
      npositions++;
    }

    // Compute centroid
    centroid = R3Centroid(npositions, positions, weights);

    // Update primitive parameters
    if ((primitive_type == R3_SEGMENTATION_NULL_PRIMITIVE_TYPE) && (npositions >= 2)) {
      RNScalar variances[3];
      R3Triad axes = R3PrincipleAxes(centroid, npositions, positions, weights, variances);
      if (variances[0] > RN_EPSILON) {
        if (variances[1] > RN_EPSILON) {
          RNScalar ratio10 = variances[1] / variances[0];
          RNScalar ratio21 = variances[2] / variances[1];
          if (ratio10 < ratio21) {
            primitive_type = R3_SEGMENTATION_LINE_PRIMITIVE_TYPE;
            line.Reset(centroid, axes[0]);
          }
          else {
            primitive_type = R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE;
            plane.Reset(centroid, axes[2]);
          }
        }
      }
    }
    else if ((primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) && (npositions >= 2)) {
      // Compute principle directions
      RNScalar variances[3];
      R3Triad axes = R3PrincipleAxes(centroid, npositions, positions, weights, variances);
      if (variances[0] > RN_EPSILON) {
        // Update line
        R3Vector direction = axes[0];
        line.Reset(centroid, direction);

        // Check if should flip line
        RNScalar dot = direction.Dot(previous_vector);
        if (dot < 0) line.Flip();
      }
    }
    else if ((primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) && (npositions >= 3)) {
      // Compute principle directions
      RNScalar variances[3];
      R3Triad axes = R3PrincipleAxes(centroid, npositions, positions, weights, variances);
      if (variances[1] > RN_EPSILON) {
        // Update plane
        R3Vector normal = axes[2];
        plane.Reset(centroid, normal);

        // Check if should flip plane
        if (seed_point) {
          RNScalar dot = normal.Dot(seed_point->normal);
          if (dot < 0) plane.Flip();
        }
        else {
          RNScalar dot = normal.Dot(previous_vector);
          if (dot < 0) plane.Flip();
        }
      }
    }

#if 0
    // Rasterize planar grid
    if (primitive_type == R3_SEGMENTATION_PLANAR_GRID_PRIMITIVE_TYPE) {
      // Rasterize points into planar grid
      R3PlanarGrid density(plane, bbox, min_cluster_spacing);
      for (int i = 0; i < points->NEntries(); i++) {
        R3SegmentationPoint *point = points->Kth(i);
        R3Point position = point->position;
        density.RasterizeWorldR3SegmentationPoint(position.X(), position.Y(), position.Z(), 1.0);
      }

      // Reset planar grid
      planar_grid.Reset(plane, bbox, min_cluster_spacing);
      R3Point seed_position = seed_point->position;
      R2R3SegmentationPoint grid_position = planar_grid.GridPosition(seed_position);
      int ix = (int) (grid_position.X() + 0.5);
      int iy = (int) (grid_position.Y() + 0.5);
      FloodCopy(density.grid, planar_grid.grid, ix, iy);
    }
#endif

    // Delete array of point positions and weights
    delete [] positions;
    delete [] weights;
  }
}



void R3SegmentationPrimitive::
Update(R3SegmentationPrimitive primitive1, R3SegmentationPrimitive primitive2, RNScalar weight1, RNScalar weight2)
{
  // Just checking
  if (weight1 == 0) {
    primitive_type = primitive2.primitive_type;
    bbox = primitive2.bbox;
    centroid = primitive2.centroid;
    line = primitive2.line;
    plane = primitive2.plane;
  }
  else if (weight2 == 0) {
    primitive_type = primitive1.primitive_type;
    bbox = primitive1.bbox;
    centroid = primitive1.centroid;
    line = primitive1.line;
    plane = primitive1.plane;
  }
  else {
    // Update primitive type
    if (primitive1.primitive_type > primitive2.primitive_type) {
      primitive_type = primitive1.primitive_type;
      weight2 = 0;
    }
    else if (primitive2.primitive_type > primitive1.primitive_type) {
      primitive_type = primitive2.primitive_type;
      weight1 = 0;
    }
    else {
      primitive_type = primitive1.primitive_type;
    }

    // Update centroid
    centroid = R3zero_point;
    centroid += weight1 * primitive1.centroid;
    centroid += weight2 * primitive2.centroid;
    centroid /= weight1 + weight2;

    // Update bbox
    bbox = R3null_box;
    bbox.Union(primitive1.bbox);
    bbox.Union(primitive2.bbox);

    // Update other stuff
    line = R3null_line;
    plane = R3null_plane;
    if (primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) {
      // Compute line
      R3Vector vector1 = primitive1.line.Vector();
      R3Vector vector2 = primitive2.line.Vector();
      if (vector1.Dot(vector2) < 0) vector2.Flip();
      R3Vector vector = R3zero_vector;
      vector += weight1 * vector1;
      vector += weight2 * vector2;
      vector /= weight1 + weight2;
      vector.Normalize();
      line.Reset(centroid, vector);
    }
    else if (primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) {
      // Compute plane
      R3Vector normal1 = primitive1.plane.Normal();
      R3Vector normal2 = primitive2.plane.Normal();
      if (normal1.Dot(normal2) < 0) normal2.Flip();
      R3Vector normal = R3zero_vector;
      normal += weight1 * normal1;
      normal += weight2 * normal2;
      normal /= weight1 + weight2;
      normal.Normalize();
      plane.Reset(centroid, normal);
    }
  }
}



////////////////////////////////////////////////////////////////////////
// R3SegmentationCluster member functions
////////////////////////////////////////////////////////////////////////

R3SegmentationCluster::
R3SegmentationCluster(R3SegmentationPoint *seed_point, int primitive_type)
  : seed_point(seed_point),
    points(),
    parent(NULL),
    children(),
    pairs(),
    primitive(primitive_type),
    area(0),
    color(0,0,0),
    timestamp(0),
    category_identifier(-1),
    category_confidence(0),
    possible_affinity(0),
    total_affinity(0),
    segmentation(NULL),
    segmentation_index(-1)
{
  // Update primitive and color and area
  if (seed_point) primitive.Update(seed_point);
  if (seed_point) color = seed_point->color;
  if (seed_point) timestamp = seed_point->timestamp;
  if (seed_point) area = seed_point->area;
  if (seed_point) category_identifier = seed_point->category_identifier;
  if (seed_point) category_confidence = seed_point->category_confidence;
}



R3SegmentationCluster::
R3SegmentationCluster(R3SegmentationPoint *seed_point, const R3SegmentationPrimitive& primitive)
  : seed_point(seed_point),
    points(),
    parent(NULL),
    children(),
    pairs(),
    primitive(primitive),
    area(0),
    color(0,0,0),
    timestamp(0),
    category_identifier(-1),
    category_confidence(0),
    possible_affinity(0),
    total_affinity(0),
    segmentation(NULL),
    segmentation_index(-1)
{
  // Update color and area
  if (seed_point) color = seed_point->color;
  if (seed_point) timestamp = seed_point->timestamp;
  if (seed_point) area = seed_point->area;
  if (seed_point) category_identifier = seed_point->category_identifier;
  if (seed_point) category_confidence = seed_point->category_confidence;
}



R3SegmentationCluster::
R3SegmentationCluster(R3SegmentationCluster *child1, R3SegmentationCluster *child2)
  : seed_point(NULL),
    points(),
    parent(NULL),
    children(),
    pairs(),
    primitive(),
    area(0),
    color(0,0,0),
    timestamp(0),
    category_identifier(-1),
    category_confidence(0),
    possible_affinity(0),
    total_affinity(0),
    segmentation(NULL),
    segmentation_index(-1)
{
  // Assign seed point
  seed_point = child1->seed_point;

  // Update primitive
  primitive.Update(child1->primitive, child2->primitive, child1->points.NEntries(), child2->points.NEntries());

  // Update color
  if (child1->points.NEntries() + child2->points.NEntries() > 0) {
    color += child1->points.NEntries() * child1->color;
    color += child2->points.NEntries() * child2->color;
    color /= child1->points.NEntries() + child2->points.NEntries();
  }

  // Update category info
  if (child1->points.NEntries() + child2->points.NEntries() > 0) {
    if (child1->category_identifier == child2->category_identifier) {
      // Categories match -- average confidences
      category_identifier = child1->category_identifier;
      category_confidence += child1->points.NEntries() * child1->category_confidence;
      category_confidence += child2->points.NEntries() * child2->category_confidence;
      category_confidence /= child1->points.NEntries() + child2->points.NEntries();
    }
    else {
      // Categories don't match -- trust larger cluster
      if ((child2->category_identifier < 0) || (child1->points.NEntries() > child2->points.NEntries())) {
        category_identifier = child1->category_identifier;
        category_confidence += child1->points.NEntries() * child1->category_confidence;
        category_confidence /= child1->points.NEntries() + child2->points.NEntries();
      }
      else {
        category_identifier = child2->category_identifier;
        category_confidence += child2->points.NEntries() * child2->category_confidence;
        category_confidence /= child1->points.NEntries() + child2->points.NEntries();
      }
    }
  }

  // Update timestamp
  timestamp = 0.5*(child1->timestamp + child2->timestamp);

  // Update area
  area = child1->area + child2->area;

  // Insert points from child1
  while (!child1->points.IsEmpty()) {
    R3SegmentationPoint *point = child1->points.Tail();
    child1->RemovePoint(point);
    RNScalar affinity = Affinity(point);
    if (affinity < 0) affinity = 0;
    possible_affinity += affinity;
    InsertPoint(point, affinity);
  }

  // Insert points from child2
  while (!child2->points.IsEmpty()) {
    R3SegmentationPoint *point = child2->points.Tail();
    child2->RemovePoint(point);
    RNScalar affinity = Affinity(point);
    if (affinity < 0) affinity = 0;
    possible_affinity += affinity;
    InsertPoint(point, affinity);
  }

  // Update hierarchy
  child1->parent = this;
  child2->parent = this;
  children.Insert(child1);
  children.Insert(child2);
}



R3SegmentationCluster::
~R3SegmentationCluster(void)
{
  // Delete children
  // for (int i = 0; i < children.NEntries(); i++) {
  //   delete children.Kth(i);
  // }

  // Remove from parent
  if (parent) parent->children.Remove(this);

  // Empty points
  EmptyPoints();
}



RNScalar R3SegmentationCluster::
Coverage(void)
{
  // Return metric of how well cluster covers points
  if (possible_affinity == 0) return 0;
  return total_affinity / possible_affinity;
}



R3Triad R3SegmentationCluster::
PrincipleAxes(R3Point *returned_center, RNScalar *returned_variances) const
{
  // Check if there are at least three points
  if (points.NEntries() < 3) {
    if (returned_center) *returned_center = primitive.centroid;
    if (returned_variances) returned_variances[0] = returned_variances[1] = returned_variances[2] = 0;
    return R3xyz_triad;
  }

  // Fill arrays of point positions and weights
  RNArray<R3Point *> positions;
  for (int i = 0; i < points.NEntries(); i++) {
    positions.Insert(&points[i]->position);
  }

  // Compute center (should be nearly same as primitive.centroid)
  R3Point center = R3Centroid(positions);
  if (returned_center) *returned_center = center;

  // Compute principle axes
  R3Triad axes = R3PrincipleAxes(center, positions, NULL, returned_variances);

  // Check if orientation is compatible with primitive
  if (primitive.primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) {
    if (primitive.line.Vector().Dot(axes[0]) < 0) axes.Reset(-axes[0], -axes[1], axes[2]);
  }
  else if (primitive.primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) {
    if (primitive.plane.Normal().Dot(axes[2]) < 0) axes.Reset(-axes[0], axes[1], -axes[2]);
  }

  // Return principle axes
  return axes;
}



R3Box R3SegmentationCluster::
AxisExtents(void)
{
  // Get centroid and axes
  R3Point center;
  R3Triad triad = PrincipleAxes(&center);
  
  // Find extent
  R3Box extent = R3null_box;
  for (int i = 0; i < points.NEntries(); i++) {
    R3Point position = points[i]->position;
    R3Vector vector = position - center;
    RNScalar x = triad.Axis(0).Dot(vector);
    RNScalar y = triad.Axis(1).Dot(vector);
    RNScalar z = triad.Axis(2).Dot(vector);
    extent.Union(R3Point(x, y, z));
  }

  // Return extent
  return extent;
}



void R3SegmentationCluster::
EmptyPoints(void)
{
  // Update points
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);
    point->cluster = NULL;
    point->cluster_affinity = 0;
    point->cluster_index = -1;
  }

  // Empty points
  points.Empty();

  // Update color and area
  color = RNblack_rgb;
  timestamp = 0;
  area = 0;

  // Update affinity
  total_affinity = 0;
}



void R3SegmentationCluster::
InsertPoint(R3SegmentationPoint *point, RNScalar affinity)
{
  // Remove from previous cluster
  if (point->cluster ) {
    if (point->cluster == this) return;
    else point->cluster->RemovePoint(point);
  }

  // Update cluster
  total_affinity += affinity; // point->cluster_affinity;
  color = (point->color + points.NEntries()*color) / (points.NEntries()+1);
  timestamp = (point->timestamp + points.NEntries()*timestamp) / (points.NEntries()+1);
  area += point->area;

  // Update category
  if ((category_identifier < 0) && (point->category_identifier >= 0)) {
    category_identifier = point->category_identifier;
    category_confidence = point->category_confidence;
  }

  // Update point
  point->cluster = this;
  point->cluster_index = points.NEntries();
  point->cluster_affinity = affinity;

  // Insert point
  points.Insert(point);
}



void R3SegmentationCluster::
RemovePoint(R3SegmentationPoint *point)
{
  // Just checking
  assert(point->cluster == this);
  assert(point->cluster_index >= 0);

  // Update cluster
  total_affinity -= point->cluster_affinity;
  color = (points.NEntries() > 1) ? (points.NEntries()*color - point->color) / (points.NEntries()-1) : RNblack_rgb;
  timestamp = (points.NEntries() > 1) ? (points.NEntries()*timestamp - point->timestamp) / (points.NEntries()-1) : 0;
  area -= point->area;

  // Remove point
  RNArrayEntry *entry = points.KthEntry(point->cluster_index);
  R3SegmentationPoint *tail = points.Tail();
  tail->cluster_index = point->cluster_index;
  points.EntryContents(entry) = tail;
  points.RemoveTail();

  // Update point
  point->cluster = NULL;
  point->cluster_index = -1;
  point->cluster_affinity = 0;
}



void R3SegmentationCluster::
InsertChild(R3SegmentationCluster *child)
{
  // Check
  assert(child != this);

  // Update area
  area += child->area;

  // Update color
  int n = points.NEntries() + child->points.NEntries();
  color *= points.NEntries();
  color += child->points.NEntries() * child->color;
  color = (n > 0) ? color / n : RNblack_rgb;

  // Update timestamp
  timestamp *= points.NEntries();
  timestamp += child->points.NEntries() * child->timestamp;
  timestamp = (n > 0) ? timestamp / n : 0;

  // Update primitive
  primitive.Update(this->primitive, child->primitive, this->points.NEntries(), child->points.NEntries());

  // Update category info
  if (points.NEntries() + child->points.NEntries() > 0) {
    if (category_identifier == child->category_identifier) {
      // Categories match -- average confidences
      category_confidence = (points.NEntries() * category_confidence + child->points.NEntries() * child->category_confidence);
      category_confidence /= points.NEntries() + child->points.NEntries();
    }
    else {
      // Categories don't match -- trust larger cluster
      if ((child->category_identifier < 0) || (points.NEntries() > child->points.NEntries())) {
        category_confidence = points.NEntries() * category_confidence;
        category_confidence /= points.NEntries() + child->points.NEntries();
      }
      else {
        category_identifier = child->category_identifier;
        category_confidence += child->points.NEntries() * child->category_confidence;
        category_confidence /= points.NEntries() + child->points.NEntries();
      }
    }
  }

  // Update affinities for current points
  if (points.NEntries() < 4 * child->points.NEntries()) {
    for (int i = 0; i < points.NEntries(); i++) {
      R3SegmentationPoint *point = points.Kth(i);
      RNScalar affinity = Affinity(point);
      if (affinity < 0) affinity = 0;
      possible_affinity += affinity - point->cluster_affinity;
      total_affinity += affinity - point->cluster_affinity;
      point->cluster_affinity = affinity;
    }
  }

  // Insert points from child
  while (!child->points.IsEmpty()) {
    R3SegmentationPoint *point = child->points.Tail();
    child->RemovePoint(point);
    RNScalar affinity = Affinity(point);
    if (affinity < 0) affinity = 0;
    possible_affinity += affinity;
    InsertPoint(point, affinity);
  }

  // Update child
  child->area = 0;
  child->color = RNblack_rgb;
  child->timestamp = 0;

  // Update hierarchy
  child->parent = this;
  children.Insert(child);
}



void R3SegmentationCluster::
RemoveChild(R3SegmentationCluster *child)
{
  // Update area
  area -= child->area;

  // Update color
  int n = points.NEntries() - child->points.NEntries();
  color *= points.NEntries();
  color -= child->points.NEntries() * child->color;
  color = (n > 0) ? color / n : RNblack_rgb;

  // Update timestamp
  timestamp *= points.NEntries();
  timestamp -= child->points.NEntries() * child->timestamp;
  timestamp = (n > 0) ? timestamp / n : 0;

  // Remove child
  this->children.Remove(child);
  child->parent = NULL;
}



int R3SegmentationCluster::
UpdatePoints(const R3Kdtree<R3SegmentationPoint *> *kdtree)
{
  // Get convenient variables
  double max_cluster_primitive_distance = (segmentation) ? segmentation->max_cluster_primitive_distance : 0;
  RNBoolean allow_outlier_points = (segmentation) ? segmentation->allow_outlier_points : TRUE;
  int min_cluster_points = (segmentation) ? segmentation->min_cluster_points : 0;
  
  // Empty points
  possible_affinity = 0;
  EmptyPoints();  

  // Find points near primitive
  if (seed_point) {
    // Find connected set of points near primitive
    static int mark = 1;
    RNArray<R3SegmentationPoint *> stack;
    InsertPoint(seed_point, 1.0);
    stack.Insert(seed_point);
    seed_point->mark = ++mark;
    while (!stack.IsEmpty()) {
      R3SegmentationPoint *point = stack.Tail();
      stack.RemoveTail();
      for (int i = 0; i < point->neighbors.NEntries(); i++) {
        R3SegmentationPoint *neighbor = point->neighbors.Kth(i);
        if (neighbor->mark == mark) continue;
        neighbor->mark = mark;
        RNScalar affinity = Affinity(neighbor);
        if (affinity <= 0) continue;
        possible_affinity += affinity;
        if (neighbor->cluster == this) continue;
        if (neighbor->cluster && (neighbor->cluster_affinity > 0.75 * affinity)) continue;
        InsertPoint(neighbor, affinity);
        stack.Insert(neighbor);
      }
    }
  }
  else if (kdtree) {
    // Find all points near primitive
    RNArray<R3SegmentationPoint *> points1;
    if (primitive.primitive_type == R3_SEGMENTATION_POINT_PRIMITIVE_TYPE)
      kdtree->FindAll(primitive.centroid, 0, max_cluster_primitive_distance, points1);
    else if (primitive.primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE)
      kdtree->FindAll(primitive.line, 0, max_cluster_primitive_distance, points1);
    else if (primitive.primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE)
      kdtree->FindAll(primitive.plane, 0, max_cluster_primitive_distance, points1);
    else RNAbort("Unrecognized primitive type");

    // Check points
    if (allow_outlier_points) {
      if (points1.NEntries() < min_cluster_points) return 0;
    }
  
    // Insert points
    for (int i = 0; i < points1.NEntries(); i++) {
      R3SegmentationPoint *point = points1.Kth(i);
      RNScalar affinity = Affinity(point);
      if (affinity <= 0) continue;
      possible_affinity += affinity;
      if (point->cluster == this) continue;
      if (point->cluster && (point->cluster->points.NEntries() > 0)) {
        RNScalar ratio =(RNScalar) points1.NEntries() / (RNScalar) point->cluster->points.NEntries();
        if (ratio < 0.1) ratio = 0.1;
        if (ratio * affinity < point->cluster_affinity) continue;
      }
      InsertPoint(point, affinity);
    }
  }

  // Return success
  return 1;
}



int R3SegmentationCluster::
UpdatePrimitive(void)
{
  // Update primitive
  primitive.Update(seed_point, &points);
  if (primitive.primitive_type == R3_SEGMENTATION_NULL_PRIMITIVE_TYPE) return 0;
  else return 1;
}



int R3SegmentationCluster::
UpdateColor(void)
{
  // Update color
  color.Reset(0,0,0);
  if (points.NEntries() == 0) return 1;
  for (int i = 0; i < points.NEntries(); i++)
    color += points[i]->color;
  color /= points.NEntries();
  return 1;
}



int R3SegmentationCluster::
UpdateTimestamp(void)
{
  // Update timestamp
  timestamp = 0;
  if (points.NEntries() == 0) return 1;
  for (int i = 0; i < points.NEntries(); i++)
    timestamp += points[i]->timestamp;
  timestamp /= points.NEntries();
  return 1;
}



int R3SegmentationCluster::
UpdateArea(void)
{
  // Update area
  area = 0;
  for (int i = 0; i < points.NEntries(); i++)
    area += points[i]->area;
  return 1;
}



static RNScalar
MergedConvexity(const R3SegmentationCluster *cluster1, const R3SegmentationCluster *cluster2)
{
  // THIS DOES NOT MAKE CLUSTERS MORE CONVEX :(
  
  int external1 = 0;
  int internal1 = 0;
  int interface1 = 0;
  int external2 = 0;
  int internal2 = 0;
  int interface2 = 0;
  int max_samples = 1024;

  int step1 = cluster1->points.NEntries() / max_samples;
  if (step1 == 0) step1 = 1;
  for (int i1 = 0; i1 < cluster1->points.NEntries(); i1 += step1) {
    R3SegmentationPoint *point1 = cluster1->points.Kth(i1);
    for (int j1 = 0; j1 < point1->neighbors.NEntries(); j1++) {
      R3SegmentationPoint *neighbor = point1->neighbors.Kth(j1);
      if (neighbor->cluster == cluster1) internal1++;
      else if (neighbor->cluster == cluster2) interface1++;
      else external1++;
    }
  }

  RNScalar numerator1 = interface1;
  RNScalar denominator1 = interface1 + internal1;
  RNScalar convexity1 = (denominator1 > 0) ? numerator1 / denominator1 : 0;
           
  int step2 = cluster2->points.NEntries() / max_samples;
  if (step2 == 0) step2 = 1;
  for (int i2 = 0; i2 < cluster2->points.NEntries(); i2 += step2) {
    R3SegmentationPoint *point2 = cluster2->points.Kth(i2);
    for (int j2 = 0; j2 < point2->neighbors.NEntries(); j2++) {
      R3SegmentationPoint *neighbor = point2->neighbors.Kth(j2);
      if (neighbor->cluster == cluster2) internal2++;
      else if (neighbor->cluster == cluster1) interface2++;
      else external2++;
    }
  }

  RNScalar numerator2 = interface2;
  RNScalar denominator2 = interface2 + internal2;
  RNScalar convexity2 = (denominator2 > 0) ? numerator2 / denominator2 : 0;
           
  return sqrt(convexity1 * convexity2);
}  



RNScalar R3SegmentationCluster::
Affinity(R3SegmentationPoint *point) const
{
  // Initialize affinity
  RNScalar affinity = 1.0;

  // Get useful variables
  R3Point position = point->position;
  double max_cluster_diameter = (segmentation) ? segmentation->max_cluster_diameter : 0;
  double max_cluster_color_difference = (segmentation) ? segmentation->max_cluster_color_difference : 0;
  double max_cluster_timestamp_difference = (segmentation) ? segmentation->max_cluster_timestamp_difference : 0;
  double max_cluster_category_difference = (segmentation) ? segmentation->max_cluster_category_difference : 0;
  double max_cluster_primitive_distance = (segmentation) ? segmentation->max_cluster_primitive_distance : 0;
  double max_cluster_normal_angle = (segmentation) ? segmentation->max_cluster_normal_angle : 0;
  RNBoolean scale_tolerances_with_depth = (segmentation) ? segmentation->scale_tolerances_with_depth : FALSE;

  // Check color difference
  if (max_cluster_color_difference > 0) {
    RNLength color_difference = 0;
    color_difference += fabs(color.R() - point->color.R());
    color_difference += fabs(color.G() - point->color.G());
    color_difference += fabs(color.B() - point->color.B());
    RNScalar color_difference_affinity = exp(color_difference * color_difference / (-2.0 * 0.25 * max_cluster_color_difference * max_cluster_color_difference));
    affinity *= color_difference_affinity;
  }

  // Check timestamp difference
  if (max_cluster_timestamp_difference > 0) {
    RNLength timestamp_difference = fabs(timestamp - point->timestamp);
    if (timestamp_difference > 0) {
      RNScalar timestamp_difference_affinity = exp(timestamp_difference * timestamp_difference / (-2.0 * max_cluster_timestamp_difference * max_cluster_timestamp_difference));
      affinity *= timestamp_difference_affinity;
    }
  }

  // Check category difference
  if (max_cluster_category_difference > 0) {
    if (category_identifier != point->category_identifier) {
      RNScalar min_category_confidence = (category_confidence <  point->category_confidence) ? category_confidence : point->category_confidence;
      RNScalar category_difference_affinity = exp(min_category_confidence * min_category_confidence / (-2.0 * max_cluster_category_difference * max_cluster_category_difference));
      affinity *= category_difference_affinity;
    }
  }

  // Check primitive distance 
  if (max_cluster_primitive_distance > 0) {
    RNLength primitive_distance = primitive.Distance(position);
    if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) primitive_distance  /= seed_point->depth;
    RNScalar primitive_distance_affinity = exp(primitive_distance * primitive_distance / (-2.0 * 0.25 * max_cluster_primitive_distance * max_cluster_primitive_distance));
    affinity *= primitive_distance_affinity;
  }

  // Check centroid distance
  if (max_cluster_diameter > 0) {
    RNLength centroid_distance = R3Distance(primitive.centroid, position);
    RNScalar centroid_distance_affinity = exp(centroid_distance * centroid_distance / (-2.0 * 0.25 * max_cluster_diameter * max_cluster_diameter));
    affinity *= centroid_distance_affinity;
  }

  // Check normal angle
  if (max_cluster_normal_angle > 0) {
    if (primitive.primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) {
      RNScalar dot = fabs(primitive.line.Vector().Dot(point->normal));
      RNAngle normal_angle = (dot < 1) ? RN_PI_OVER_TWO - acos(dot) : RN_PI_OVER_TWO;
      if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) normal_angle /= seed_point->depth;
      RNScalar normal_angle_affinity = exp(normal_angle * normal_angle / (-2.0 * 0.25 * max_cluster_normal_angle * max_cluster_normal_angle));
      affinity *= normal_angle_affinity;
    }
    else if (primitive.primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) {
      RNScalar dot = primitive.plane.Normal().Dot(point->normal);
      RNAngle normal_angle = (dot > -1) ? ((dot < 1) ? acos(dot) : 0) : RN_PI;
      if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) normal_angle /= seed_point->depth;
      RNScalar normal_angle_affinity = exp(normal_angle * normal_angle / (-2.0 * 0.25 * max_cluster_normal_angle * max_cluster_normal_angle));
      affinity *= normal_angle_affinity;
    }
  }

  // Just checking
  assert(affinity >= 0);

  // Return affinity
  return affinity;
}



RNScalar R3SegmentationCluster::
Affinity(R3SegmentationCluster *cluster) const
{
  // Get convenient variables
  R3Segmentation *segmentation = cluster->segmentation;
  double max_pair_color_difference = (segmentation) ? segmentation->max_pair_color_difference : 0;
  double max_pair_timestamp_difference = (segmentation) ? segmentation->max_pair_timestamp_difference : 0;
  double max_pair_category_difference = (segmentation) ? segmentation->max_pair_category_difference : 0;
  double max_pair_centroid_distance = (segmentation) ? segmentation->max_pair_centroid_distance : 0;
  double max_pair_primitive_distance = (segmentation) ? segmentation->max_pair_primitive_distance : 0;
  double max_pair_normal_angle = (segmentation) ? segmentation->max_pair_normal_angle : 0;
  RNBoolean equalize_cluster_sizes = (segmentation) ? segmentation->equalize_cluster_sizes : FALSE;
  RNBoolean balance_cluster_sizes = (segmentation) ? segmentation->balance_cluster_sizes : FALSE;
  RNBoolean favor_convex_clusters = (segmentation) ? segmentation->favor_convex_clusters : FALSE;
  RNBoolean scale_tolerances_with_depth = (segmentation) ? segmentation->scale_tolerances_with_depth : FALSE;

  // Initialize affinity
  RNScalar affinity = 1;

  // Check color difference
  if (max_pair_color_difference > 0) {
    RNLength color_difference = 0;
    color_difference += fabs(color.R() - cluster->color.R());
    color_difference += fabs(color.G() - cluster->color.G());
    color_difference += fabs(color.B() - cluster->color.B());
    RNScalar color_difference_affinity = exp(color_difference * color_difference / (-2.0 * 0.25 * max_pair_color_difference * max_pair_color_difference));
    affinity *= color_difference_affinity;
    assert(affinity >= 0);
  }

  // Check timestamp difference
  if (max_pair_timestamp_difference > 0) {
    RNLength timestamp_difference = fabs(timestamp - cluster->timestamp);
    if (timestamp_difference > 0) {
      RNScalar timestamp_difference_affinity = exp(timestamp_difference * timestamp_difference / (-2.0 *  max_pair_timestamp_difference * max_pair_timestamp_difference));
      affinity *= timestamp_difference_affinity;
      assert(affinity >= 0);
    }
  }

  // Check category difference
  if (max_pair_category_difference > 0) {
    if (category_identifier != cluster->category_identifier) {
      RNScalar min_category_confidence = (category_confidence <  cluster->category_confidence) ? category_confidence : cluster->category_confidence;
      RNScalar category_difference_affinity = exp(min_category_confidence * min_category_confidence / (-2.0 * max_pair_category_difference * max_pair_category_difference));
      affinity *= category_difference_affinity;
    }
  }

  // Compute centroid distance
  if (max_pair_centroid_distance > 0) {
    RNLength centroid_distance = R3Distance(primitive.centroid, cluster->primitive.centroid);
    RNScalar centroid_distance_affinity = exp(centroid_distance * centroid_distance / (-2.0 * 0.25 * max_pair_centroid_distance * max_pair_centroid_distance));
    affinity *= centroid_distance_affinity;
    assert(affinity >= 0);
  }

  // Compute primitive distances
  if (max_pair_primitive_distance > 0) {
    // Compute point0-primitive1 distance
    RNLength primitive0_distance = primitive.Distance(cluster->primitive.centroid);
    if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) primitive0_distance  /= seed_point->depth;
    RNScalar primitive0_distance_affinity = exp(primitive0_distance * primitive0_distance / (-2.0 * 0.25 * max_pair_primitive_distance * max_pair_primitive_distance));
    affinity *= primitive0_distance_affinity;
    assert(affinity >= 0);

    // Compute point1-primitive0 distance
    RNLength primitive1_distance = cluster->primitive.Distance(primitive.centroid);
    if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) primitive1_distance  /= seed_point->depth;
    RNScalar primitive1_distance_affinity = exp(primitive1_distance * primitive1_distance / (-2.0 * 0.25 * max_pair_primitive_distance * max_pair_primitive_distance));
    affinity *= primitive1_distance_affinity;
    assert(affinity >= 0);
  }

  // Compute normal angle
  if (max_pair_normal_angle > 0) {
    if ((primitive.primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) &&
      (cluster->primitive.primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE)) {
      RNScalar dot = fabs(primitive.line.Vector().Dot(cluster->primitive.line.Vector()));
      RNAngle normal_angle = (dot < 1) ? acos(dot) : 0;
      if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) normal_angle  /= seed_point->depth;
      RNScalar normal_angle_affinity = exp(normal_angle * normal_angle / (-2.0 * 0.25 * max_pair_normal_angle * max_pair_normal_angle));
      affinity *= normal_angle_affinity;
      assert(affinity >= 0);
    }
    else if ((primitive.primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) &&
      (cluster->primitive.primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE)) {
      RNScalar dot = fabs(primitive.plane.Normal().Dot(cluster->primitive.line.Vector()));
      RNAngle normal_angle = (dot < 1) ? RN_PI_OVER_TWO - acos(dot) : RN_PI_OVER_TWO;
      if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) normal_angle  /= seed_point->depth;
      RNScalar normal_angle_affinity = exp(normal_angle * normal_angle / (-2.0 * 0.25 * max_pair_normal_angle * max_pair_normal_angle));
      affinity *= normal_angle_affinity;
      assert(affinity >= 0);
    }
    else if ((primitive.primitive_type == R3_SEGMENTATION_LINE_PRIMITIVE_TYPE) &&
      (cluster->primitive.primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE)) {
      RNScalar dot = fabs(primitive.line.Vector().Dot(cluster->primitive.plane.Normal()));
      RNAngle normal_angle = (dot < 1) ? RN_PI_OVER_TWO - acos(dot) : RN_PI_OVER_TWO;
      if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) normal_angle  /= seed_point->depth;
      RNScalar normal_angle_affinity = exp(normal_angle * normal_angle / (-2.0 * 0.25 * max_pair_normal_angle * max_pair_normal_angle));
      affinity *= normal_angle_affinity;
      assert(affinity >= 0);
    }
    else if ((primitive.primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE) &&
      (cluster->primitive.primitive_type == R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE)) {
      RNScalar dot = primitive.plane.Normal().Dot(cluster->primitive.plane.Normal());
      RNAngle normal_angle = (dot > -1) ? ((dot < 1) ? acos(dot) : 0) : RN_PI;
      if (seed_point && (seed_point->depth > 0) && scale_tolerances_with_depth) normal_angle  /= seed_point->depth;
      RNScalar normal_angle_affinity = exp(normal_angle * normal_angle / (-2.0 * 0.25 * max_pair_normal_angle * max_pair_normal_angle));
      affinity *= normal_angle_affinity;
      assert(affinity >= 0);
    }
  }

  // Compute fraction of points in smallest cluster
  if (equalize_cluster_sizes) {
    int segmentation_npoints = (segmentation) ? segmentation->points.NEntries() : 0;
    int segmentation_nclusters = (segmentation) ? segmentation->clusters.NEntries() : 0;
    if ((segmentation_npoints > 0) && (segmentation_nclusters > 0)) {
      int smaller_npoints = (points.NEntries() < cluster->points.NEntries()) ? points.NEntries() : cluster->points.NEntries();
      if (smaller_npoints == 0) return 0;
      RNScalar avg_points_per_cluster = (RNScalar) segmentation_npoints / (RNScalar) segmentation_nclusters;
      RNScalar balance_affinity =  avg_points_per_cluster / smaller_npoints;
      affinity *= balance_affinity;
      assert(affinity >= 0);
    }
  }
  
  // Compute relative sizes of clusters
  if (balance_cluster_sizes) {
    RNScalar balance_affinity = 1.0;
    if (points.NEntries() == 0) return 0;
    if (cluster->points.NEntries() == 0) return 0;
    if (points.NEntries() < cluster->points.NEntries()) balance_affinity = (RNScalar) points.NEntries() / (RNScalar) cluster->points.NEntries();
    else balance_affinity = (RNScalar) cluster->points.NEntries() / (RNScalar) points.NEntries();
    affinity *= balance_affinity;
    assert(affinity >= 0);
  }
  
  // Compute fraction of points in smallest cluster
  if (favor_convex_clusters) {
    RNScalar convexity_affinity = MergedConvexity(this, cluster);
    affinity *= convexity_affinity;
    assert(affinity >= 0);
  }
  
  // Just checking
  assert(affinity >= 0);

  // Return affinity
  return affinity;
}



static int
CompareClusters(const void *data1, const void *data2)
{
  R3SegmentationCluster *cluster1 = *((R3SegmentationCluster **) data1);
  R3SegmentationCluster *cluster2 = *((R3SegmentationCluster **) data2);
  if (cluster2->total_affinity > cluster1->total_affinity) return 1;
  else if (cluster1->total_affinity > cluster2->total_affinity) return -1;
  else return 0;
}



////////////////////////////////////////////////////////////////////////
// R3SegmentationPair member functions
////////////////////////////////////////////////////////////////////////

R3SegmentationPair::
R3SegmentationPair(R3SegmentationCluster *cluster1, R3SegmentationCluster *cluster2, RNScalar affinity)
  : affinity(affinity),
    heapentry(NULL)
{
  // Insert pair into clusters
  if (cluster1 && cluster2) {
    // Remember clusters
    clusters[0] = cluster1;
    clusters[1] = cluster2;

    // Remember position of pair in clusters
    cluster_index[0] = cluster1->pairs.NEntries();
    cluster_index[1] = cluster2->pairs.NEntries();

    // Update clusters
    cluster1->pairs.Insert(this);
    cluster2->pairs.Insert(this);
  }
  else {
    // Initialize clusters
    clusters[0] = NULL;
    clusters[1] = NULL;

    // Initialize cluster index
    cluster_index[0] = -1;
    cluster_index[1] = -1;
  }
}



R3SegmentationPair::
~R3SegmentationPair(void)
{
  // Remove this pair from first cluster
  if (clusters[0]) {
    assert(cluster_index[0] >= 0);
    RNArrayEntry *entry = clusters[0]->pairs.KthEntry(cluster_index[0]);
    R3SegmentationPair *tail = clusters[0]->pairs.Tail();
    if (tail->clusters[0] == clusters[0]) tail->cluster_index[0] = cluster_index[0];
    else if (tail->clusters[1] == clusters[0]) tail->cluster_index[1] = cluster_index[0];
    clusters[0]->pairs.EntryContents(entry) = tail;
    clusters[0]->pairs.RemoveTail();
  }

  // Remove this pair from second cluster
  if (clusters[1]) {
    assert(cluster_index[1] >= 0);
    RNArrayEntry *entry = clusters[1]->pairs.KthEntry(cluster_index[1]);
    R3SegmentationPair *tail = clusters[1]->pairs.Tail();
    if (tail->clusters[0] == clusters[1]) tail->cluster_index[0] = cluster_index[1];
    else if (tail->clusters[1] == clusters[1]) tail->cluster_index[1] = cluster_index[1];
    clusters[1]->pairs.EntryContents(entry) = tail;
    clusters[1]->pairs.RemoveTail();
  }
}



static R3SegmentationPair *
FindR3SegmentationPair(R3SegmentationCluster *cluster1, R3SegmentationCluster *cluster2) 
{
  // Swap clusters so that cluster1 has fewer pairs
  if (cluster1->pairs.NEntries() > cluster2->pairs.NEntries()) {
    R3SegmentationCluster *swap = cluster1; 
    cluster1 = cluster2; 
    cluster2 = swap;
  }

  // Search for pair
  for (int i = 0; i < cluster1->pairs.NEntries(); i++) {
    R3SegmentationPair *pair = cluster1->pairs.Kth(i);
    if (pair->clusters[0] == cluster2) return pair;
    if (pair->clusters[1] == cluster2) return pair;
  }

  // R3SegmentationPair not found
  return NULL;
}



////////////////////////////////////////////////////////////////////////
// R3Segmentation functions
////////////////////////////////////////////////////////////////////////

R3Segmentation::
R3Segmentation(void)
  : points(),
    kdtree(NULL),
    clusters(),
    point_buffer(NULL),
    min_clusters(0),
    max_clusters(0),
    min_cluster_points(0),
    min_cluster_area(0),
    min_cluster_coverage(0),
    max_cluster_diameter(16),
    max_cluster_primitive_distance(0.01),
    max_cluster_normal_angle(RN_PI / 4.0),
    max_cluster_color_difference(0.5),
    max_cluster_timestamp_difference(0),
    max_cluster_category_difference(0),
    max_pair_centroid_distance(16),
    max_pair_primitive_distance(0.01),
    max_pair_normal_angle(RN_PI / 4.0),
    max_pair_color_difference(0.5),
    max_pair_timestamp_difference(0),
    max_pair_category_difference(0),
    min_pair_affinity(RN_EPSILON),
    max_refinement_iterations(3),
    max_reassignment_iterations(0),
    equalize_cluster_sizes(TRUE),
    balance_cluster_sizes(TRUE),
    favor_convex_clusters(FALSE),
    scale_tolerances_with_depth(FALSE),
    initialize_hierarchically(TRUE),
    allow_outlier_points(FALSE),
    refine_boundaries(TRUE),
    print_progress(FALSE)
{
}



R3Segmentation::
R3Segmentation(const R3Segmentation& segmentation)
  : points(),
    kdtree(NULL),
    clusters(),
    point_buffer(NULL),
    min_clusters(segmentation.min_clusters),
    max_clusters(segmentation.max_clusters),
    min_cluster_points(segmentation.min_cluster_points),
    min_cluster_area(segmentation.min_cluster_area),
    min_cluster_coverage(segmentation.min_cluster_coverage),
    max_cluster_diameter(segmentation.max_cluster_diameter),
    max_cluster_primitive_distance(segmentation.max_cluster_primitive_distance),
    max_cluster_normal_angle(segmentation.max_cluster_normal_angle),
    max_cluster_color_difference(segmentation.max_cluster_color_difference),
    max_cluster_timestamp_difference(segmentation.max_cluster_timestamp_difference),
    max_cluster_category_difference(segmentation.max_cluster_category_difference),
    max_pair_centroid_distance(segmentation.max_pair_centroid_distance),
    max_pair_primitive_distance(segmentation.max_pair_primitive_distance),
    max_pair_normal_angle(segmentation.max_pair_normal_angle),
    max_pair_color_difference(segmentation.max_pair_color_difference),
    max_pair_timestamp_difference(segmentation.max_pair_timestamp_difference),
    max_pair_category_difference(segmentation.max_pair_category_difference),
    min_pair_affinity(segmentation.min_pair_affinity),
    max_refinement_iterations(segmentation.max_refinement_iterations),
    max_reassignment_iterations(segmentation.max_reassignment_iterations),
    equalize_cluster_sizes(segmentation.equalize_cluster_sizes),
    balance_cluster_sizes(segmentation.balance_cluster_sizes),
    favor_convex_clusters(segmentation.favor_convex_clusters),
    scale_tolerances_with_depth(segmentation.scale_tolerances_with_depth),
    initialize_hierarchically(segmentation.initialize_hierarchically),
    allow_outlier_points(segmentation.allow_outlier_points),
    refine_boundaries(segmentation.refine_boundaries),
    print_progress(segmentation.print_progress)
{
  // Not implemented, unless segmentation is empty
  assert(points.IsEmpty());
  assert(clusters.IsEmpty());
}



R3Segmentation::
~R3Segmentation(void)
{
  // Delete clusters
  for (int i = 0; i < clusters.NEntries(); i++) delete clusters[i];
  
  // Delete kdtree
  if (kdtree) delete kdtree;

  // Delete points
  if (point_buffer) delete [] point_buffer;
  else { for (int i = 0; i < points.NEntries(); i++) delete points[i]; }
}



RNScalar R3Segmentation::
Affinity(void) const
{
  RNScalar sum = 0;
  for (int i = 0; i < clusters.NEntries(); i++) {
    sum += clusters[i]->total_affinity;
  }
  return sum;
}



int R3Segmentation::
NUnclusteredPoints(void) const
{
  // Count unclustered points
  int count = 0;
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);
    if (!point->cluster) count++;
  }

  // Return number of unclustered points
  return count;
}



RNScalar R3Segmentation::
AverageNeighborCount(void) const
{
  // Compute count
  int count = points.NEntries();
  if (count == 0) return 0;

  // Compute sum
  int sum = 0;
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);
    sum += point->neighbors.NEntries();
  }

  // Return mean
  return (double) sum / (double) count;
}



static RNScalar
R3SegmentationPointAffinity(
  const R3SegmentationPoint *point1, const R3SegmentationPoint *point2,
  double max_distance,
  double max_primitive_distance,
  double max_normal_angle,
  double max_color_difference,
  double max_timestamp_difference,
  double max_category_difference,
  double min_affinity = 0)
{
  // Initialize affinity
  RNScalar affinity = 1.0;

  // Check category difference
  if (max_category_difference > 0) {
    if (point1->category_identifier != point2->category_identifier) {
      RNScalar max_d = max_category_difference;
      RNScalar d = (point1->category_confidence <  point2->category_confidence) ? point1->category_confidence : point2->category_confidence;
      if (d > max_d) return 0;
      affinity *= 1.0 - (d/max_d);
      // affinity *= exp(d*d / (-2.0*max_d*max_d));
      if (affinity < min_affinity) return 0;
    }
  }

  // Check timestamp difference
  if (max_timestamp_difference > 0) {
    RNScalar max_d = max_timestamp_difference;
    RNLength d = fabs(point1->timestamp - point2->timestamp);
    if (d > max_d) return 0;
    // affinity *= exp(d*d / (-2.0*max_d*max_d));
    affinity *= 1.0 - (d/max_d);
    if (affinity < min_affinity) return 0;
  }

  // Check distance
  if (max_distance > 0) {
    RNScalar max_dd = max_distance * max_distance;
    RNScalar dd = R3SquaredDistance(point1->position, point2->position);
    if (dd > max_dd) return 0;
    // affinity *= exp(dd / (-2.0 * max_dd));
    affinity *= 1.0 - (dd/max_dd);
    if (affinity < min_affinity) return 0;
  }

  // Check primitive distance 
  if (max_primitive_distance > 0) {
    RNScalar max_d = max_primitive_distance;
    RNLength d1 = R3Distance(R3Plane(point1->position, point1->normal), point2->position);
    if (d1 > max_d) return 0;
    RNLength d2 = R3Distance(R3Plane(point2->position, point2->normal), point1->position);
    if (d2 > max_d) return 0;
    // affinity *= exp(d1*d1 / (-2.0 * 0.25 * max_d * max_d));
    // affinity *= exp(d2*d2 / (-2.0 * 0.25 * max_d * max_d));
    affinity *= 1.0 - (d1/max_d);
    affinity *= 1.0 - (d2/max_d);
    if (affinity < min_affinity) return 0;
  }

  // Check normal angle
  if (max_normal_angle > 0) {
    RNScalar max_a = max_normal_angle;
    RNScalar dot = point1->normal.Dot(point2->normal);
    RNAngle a = (dot > -1) ? ((dot < 1) ? acos(dot) : 0) : RN_PI;
    if (a > max_a) return 0;
    // affinity *= exp(a*a / (-2.0 * 0.25 * max_a * max_a));
    affinity *= 1.0 - (a/max_a);
    if (affinity < min_affinity) return 0;
  }

  // Check color difference
  if (max_color_difference > 0) {
    RNScalar max_d = max_color_difference;
    RNLength d = 0;
    d += fabs(point1->color.R() - point2->color.R());
    d += fabs(point1->color.G() - point2->color.G());
    d += fabs(point1->color.B() - point2->color.B());
    if (d > max_d) return 0;
    // affinity *= exp(d*d / (-2.0*0.25*max_d*max_d));
    affinity *= 1.0 - (d/max_d);
    if (affinity < min_affinity) return 0;
  }

  // Just checking
  assert(affinity <= 1);
  assert(affinity >= 0);

  // Return affinity
  return affinity;
}



#if 0

int R3Segmentation::
CreateNeighbors(
  int max_neighbor_count,
  double max_neighbor_distance,
  double max_neighbor_primitive_distance,
  double max_neighbor_normal_angle,
  double max_neighbor_color_difference,
  double max_neighbor_distance_factor,
  double max_neighbor_timestamp_difference,
  double max_neighbor_category_difference,
  RNBoolean partition_identifiers)
{
  // Compute bounding box
  R3Box bbox = R3null_box;
  for (int i = 0; i < points.NEntries(); i++) {
    bbox.Union(points[i]->position);
  }

  // Compute max neighbor distance
  double max_length = bbox.LongestAxisLength();
  if (RNIsZero(max_length)) return 1;
  if (max_neighbor_distance == 0) max_neighbor_distance = 0.01 * max_length;

  // Compute bbox dimensions
  double xmin = bbox.XMin();
  double ymin = bbox.YMin();
  double zmin = bbox.ZMin();
  double xlength = bbox.XLength() + RN_EPSILON;
  double ylength = bbox.YLength() + RN_EPSILON;
  double zlength = bbox.ZLength() + RN_EPSILON;
  if (xlength < max_neighbor_distance) xlength = max_neighbor_distance;
  if (ylength < max_neighbor_distance) ylength = max_neighbor_distance;
  if (zlength < max_neighbor_distance) zlength = max_neighbor_distance;

  // Compute grid resolution
  int xres = 2 * xlength / max_neighbor_distance + RN_EPSILON;
  int yres = 2 * ylength / max_neighbor_distance + RN_EPSILON;
  int zres = 2 * zlength / max_neighbor_distance + RN_EPSILON;
  if (xres > 512) xres = 512;
  if (yres > 512) yres = 512;
  if (zres > 512) zres = 512;

  // Compute world-to-grid scale factor
  double xscale = xres / xlength;
  double yscale = yres / ylength;
  double zscale = zres / zlength;
  
  // Create cells with lists of points
  int ncells = xres * yres * zres;
  RNArray<R3SegmentationPoint *> *cells = new RNArray<R3SegmentationPoint *> [ ncells ];
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);
    const R3Point& p = point->position;
    int ix = xscale * (p.X() - xmin);
    int iy = yscale * (p.Y() - ymin);
    int iz = zscale * (p.Z() - zmin);
    assert((ix >= 0) && (ix < xres));
    assert((iy >= 0) && (iy < yres));
    assert((iz >= 0) && (iz < zres));
    cells[iz*xres*yres + iy*xres + ix].Insert(point);
  }

  // Create neighbors
  for (int cz = 0; cz < zres; cz++ ) {
    for (int cy = 0; cy < yres; cy++ ) {
      for (int cx = 0; cx < xres; cx++ ) {
        // Get array of points
        const RNArray<R3SegmentationPoint *>& cell_points = cells[cz*xres*yres + cy*xres + cx];
        if (cell_points.IsEmpty()) continue;
        double radius_sum = 0;
        int cell_count = 0;
        int point_count = 0;
        int neighbor_count = 0;
        
        // Create neighbors for each point
        for (int i = 0; i < cell_points.NEntries(); i++) {
          R3SegmentationPoint *point = cell_points.Kth(i);

          // Refine maximum distance for this point
          RNScalar max_d = max_neighbor_distance;
          if ((max_neighbor_distance_factor > 0) && (point->radius1 > 0)) 
             max_d = max_neighbor_distance_factor * point->radius1;
          if (max_d == 0) max_d = 10 * point->radius1;
          if (max_d == 0) max_d = 1;
          radius_sum += max_d;
          
          // Compute search window based on max_d
          int max_dz = zscale*max_d + 1;
          int max_dy = yscale*max_d + 1;
          int max_dx = xscale*max_d + 1;

          // Find neighbor points
          // by searching neighbor cells (closest to furthest)
          double min_affinity = 0;
          for (int dz = 0; dz <= max_dz; dz++) {
            for (int sz = -1; sz <= 1; sz += 2) {
              if ((dz == 0) && (sz == -1)) continue;
              int iz = cz + sz*dz;
              if ((iz < 0) || (iz >= zres)) continue;
              for (int dy = 0; dy <= max_dy; dy++) {
                for (int sy = -1; sy <= 1; sy += 2) {
                  if ((dy == 0) && (sy == -1)) continue;
                  int iy = cy + sy*dy;
                  if ((iy < 0) || (iy >= yres)) continue;
                  for (int dx = 0; dx <= max_dx; dx++) {
                    for (int sx = -1; sx <= 1; sx += 2) {
                      if ((dx == 0) && (sx == -1)) continue;
                      int ix = cx + sx*dx;
                      if ((ix < 0) || (ix >= xres)) continue;

                      // Get points from cell
                      const RNArray<R3SegmentationPoint *> &neighbors = cells[iz*xres*yres + iy*xres + ix];
                      cell_count++;

                      // Find neighbors with highest affinity
                      for (int j = 0; j < neighbors.NEntries(); j++) {
                        R3SegmentationPoint *neighbor = neighbors.Kth(j);
                        point_count++;

                        // Check identifier
                        if (partition_identifiers) {
                          if (point->identifier != neighbor->identifier) continue;
                        }

                        // Compute affinity
                        RNScalar affinity = R3SegmentationPointAffinity(point, neighbor,
                          max_d, max_neighbor_primitive_distance,
                          max_neighbor_normal_angle, max_neighbor_color_difference,
                          max_neighbor_timestamp_difference,
                          max_neighbor_category_difference, min_affinity);
                        if (affinity <= min_affinity) continue;
                        neighbor_count++;
                        
                        // Insert neighbor
                        if (max_neighbor_count > 0) {
                          // Find slot for neighbor in list sorted by affinity
                          int slot = point->neighbors.NEntries();
                          while (slot > 0) {
                            RNScalar s_affinity = point->neighbors[slot-1]->cluster_affinity;
                            if (s_affinity > affinity) break;
                            slot--;
                          }
                          if (slot < max_neighbor_count) {
                            neighbor->cluster_affinity = affinity;
                            point->neighbors.InsertKth(neighbor, slot);
                            point->neighbors.Truncate(max_neighbor_count);
                          }
                          if (point->neighbors.NEntries() == max_neighbor_count) {
                            min_affinity = point->neighbors[point->neighbors.NEntries()-1]->cluster_affinity;
                          }
                        }
                        else {
                          // Insert neighbor at end of growing list
                          point->neighbors.Insert(neighbor);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }

#if 0
        if (!cell_points.IsEmpty()) {
          printf("%d %d %d/%d: %d : %8.2f %8.2f %8.2f %8.2f\n",
            cx, cy, cz, zres, cell_points.NEntries(),
            radius_sum / (double) cell_points.NEntries(),
            (double) cell_count / (double) cell_points.NEntries(),
            (double) point_count / (double) cell_points.NEntries(),
            (double) neighbor_count / (double) cell_points.NEntries());
        }
#endif
      }
    }
  }

  // Delete cells
  delete [] cells;

  // Reset cluster affinities
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);
    point->cluster_affinity = 0;
  }
  
  // Return success
  return 1;
}

#else

int R3Segmentation::
CreateNeighbors(
  int max_neighbor_count,
  double max_neighbor_distance,
  double max_neighbor_primitive_distance,
  double max_neighbor_normal_angle,
  double max_neighbor_color_difference,
  double max_neighbor_distance_factor,
  double max_neighbor_timestamp_difference,
  double max_neighbor_category_difference,
  RNBoolean partition_identifiers)
{
  // Create kdtree of points
  R3SegmentationPoint tmp; int position_offset = (unsigned char *) &(tmp.position) - (unsigned char *) &tmp;
  kdtree = new R3Kdtree<R3SegmentationPoint *>(points, position_offset);
  if (!kdtree) {
    RNFail("Unable to create kdtree\n");
    return 0;
  }
  
  // Create arrays of neighbor points
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);

    // Refine maximum distance for this point
    RNScalar max_d = max_neighbor_distance;
    if ((max_neighbor_distance_factor > 0) && (point->radius1 > 0)) 
      max_d = max_neighbor_distance_factor * point->radius1;
    if (max_d == 0) max_d = 10 * point->radius1;
    if (max_d == 0) max_d = 1;

    // Create neighbors
    double min_affinity = 0;
    RNArray<R3SegmentationPoint *> neighbors;
    if (kdtree->FindClosest(point, 0, max_d, max_neighbor_count, neighbors)) {
      for (int j = 0; j < neighbors.NEntries(); j++) {
        R3SegmentationPoint *neighbor = neighbors.Kth(j);
        if (neighbor == point) continue;
        
        // Check identifier
        if (partition_identifiers) {
          if (point->identifier != neighbor->identifier) continue;
        }

        // Compute affinity
        RNScalar affinity = R3SegmentationPointAffinity(point, neighbor,
          max_d, max_neighbor_primitive_distance,
          max_neighbor_normal_angle, max_neighbor_color_difference,
          max_neighbor_timestamp_difference,
          max_neighbor_category_difference, min_affinity);
        if (affinity <= min_affinity) continue;
        
        // Insert neighbor
        if ((max_neighbor_count > 0) &&
            (neighbors.NEntries() - j > max_neighbor_count - point->neighbors.NEntries())) {
          // Find slot for neighbor in list sorted by affinity
          int slot = point->neighbors.NEntries();
          while (slot > 0) {
            RNScalar s_affinity = point->neighbors[slot-1]->cluster_affinity;
            if (s_affinity > affinity) break;
            slot--;
          }
          if (slot < max_neighbor_count) {
            neighbor->cluster_affinity = affinity;
            point->neighbors.InsertKth(neighbor, slot);
            point->neighbors.Truncate(max_neighbor_count);
          }
          if (point->neighbors.NEntries() == max_neighbor_count) {
            min_affinity = point->neighbors[point->neighbors.NEntries()-1]->cluster_affinity;
          }
        }
        else {
          // Insert neighbor at end of unconstrained list
          point->neighbors.Insert(neighbor);
        }
      }
    }
  }

  // Reset cluster affinities
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);
    point->cluster_affinity = 0;
  }
  
  // Return success
  return 1;
}

#endif



int R3Segmentation::
UpdatePoints(void)
{
  // Update normal, tangent, radii, and area from neighbors
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);
    if (point->neighbors.NEntries() < 2) continue;
    if (!point->normal.IsZero() && !point->tangent.IsZero() &&
        (point->radius1 > 0) && (point->radius2 > 0)) continue;

    // Create array of neighborhood positions
    RNArray<R3Point *> positions;
    positions.Insert(&point->position);
    for (int j = 0; j < point->neighbors.NEntries(); j++) {
      positions.Insert(&point->neighbors[j]->position);
    }

    // Compute principle axes
    RNScalar variances[3];
    R3Point centroid = R3Centroid(positions);
    R3Triad axes = R3PrincipleAxes(centroid, positions, NULL, variances);
    if (RNIsZero(variances[1])) continue;

    // Update point
    point->normal = axes[2];
    point->tangent = axes[0];
    point->radius1 = sqrt(variances[0]);
    point->radius2 = sqrt(variances[1]);
    point->area = point->radius1 * point->radius2;
  }

  // Return success
  return 1;
}



int R3Segmentation::
CreateSingletonClusters(int primitive_type)
{
  // Create cluster for every point
  for (int i = 0; i < points.NEntries(); i++) {
    R3SegmentationPoint *point = points.Kth(i);

    // Create primitive
    R3SegmentationPrimitive primitive(primitive_type);
    primitive.Update(point);
    
    // Create cluster
    R3SegmentationCluster *cluster = new R3SegmentationCluster(point, primitive);

    // Insert point
    cluster->InsertPoint(point, 1.0);

    // Insert cluster
    cluster->segmentation = this;
    cluster->segmentation_index = clusters.NEntries();    
    clusters.Insert(cluster);
  }

  // Return success
  return 1;
}



int R3Segmentation::
CreateRegionGrowingClusters(int primitive_type)
{
  // Determine how many seed points to skip each iteration
  int skip = 1;
  if (allow_outlier_points) {
    if ((max_clusters > 0) && (points.NEntries()/(4*max_clusters) > skip))
      skip = points.NEntries()/(4*max_clusters);
    if ((min_cluster_points > 0) && ((min_cluster_points/4) > skip))
      skip = min_cluster_points/4;
    if ((min_clusters > 0) && (skip > points.NEntries()/min_clusters))
      skip = points.NEntries()/min_clusters;
  }

  // Search seed points
  int seed_index = 0;
  while (seed_index < points.NEntries()) {
    // Find next seed point
    R3SegmentationPoint *seed_point = NULL;
    while ((seed_index < points.NEntries()) && !seed_point) {
      R3SegmentationPoint *point = points.Kth(seed_index);
      if (!point->cluster) seed_point = point;
      seed_index += skip; 
    }

    // Check seed point
    if (!seed_point) break;

    // Create cluster
    R3SegmentationPrimitive primitive(primitive_type);
    primitive.Update(seed_point);
    R3SegmentationCluster *cluster = new R3SegmentationCluster(seed_point, primitive);
    if (!cluster->UpdatePoints(kdtree)) { delete cluster; continue; }
    if (!cluster->UpdateColor()) { delete cluster; continue; }
    if (!cluster->UpdateTimestamp()) { delete cluster; continue; }

    // Insert cluster
    cluster->segmentation = this;
    cluster->segmentation_index = clusters.NEntries();    
    clusters.Insert(cluster);
  } 

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// R3SegmentationClustering manipulation functions
////////////////////////////////////////////////////////////////////////

int R3Segmentation::
ReassignClusters(void)
{
  // Iteratively update everything
  for (int iter = 0; iter < max_reassignment_iterations; iter++) {
    // Copy list of clusters
    RNArray<R3SegmentationCluster *> tmp = clusters;
    tmp.Sort(CompareClusters);

    // Will rebuild list of clusters
    clusters.Empty();

    // Update each cluster
    RNBoolean converged = TRUE;
    for (int i = 0; i < tmp.NEntries(); i++) {
      R3SegmentationCluster *cluster = tmp.Kth(i);
      int prev_npoints = cluster->points.NEntries();

      // Update cluster (including reassigning points)
      RNBoolean error = FALSE;
      if (!error && !cluster->UpdatePrimitive()) error = TRUE;
      if (!error && !cluster->UpdatePoints(kdtree)) error = TRUE; 
      if (!error && !cluster->UpdateColor()) error = TRUE; 
      if (!error && !cluster->UpdateTimestamp()) error = TRUE; 

      // Insert cluster
      cluster->segmentation = this;
      cluster->segmentation_index = clusters.NEntries();    
      if (!error) clusters.Insert(cluster);
      else delete cluster;

      // Check for convergence
      if (error || (prev_npoints != cluster->points.NEntries())) {
        converged = FALSE;
      }
    }

    // Check if converged
    if (converged) break;
  }

  // Return success
  return 1;
}



int R3Segmentation::
DeleteClusters(void)
{
  // Check if should delete clusters
  if (!allow_outlier_points) return 1;
  
  // Sort clusters
  clusters.Sort(CompareClusters);

  // Separate viable from nonviable ones
  RNArray<R3SegmentationCluster *> viable_clusters;
  RNArray<R3SegmentationCluster *> nonviable_clusters;
  for (int i = 0; i < clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = clusters.Kth(i);

    // Check min_clusters
    if ((min_clusters <= 0) || (i >= min_clusters)) {
      // Check cluster points
      if (min_cluster_points > 0) {
        if (cluster->points.NEntries() < min_cluster_points) {
          nonviable_clusters.Insert(cluster);
          continue;
        }
      }

      // Check cluster area
      if (min_cluster_area > 0) {
        if (cluster->area < min_cluster_area) {
          nonviable_clusters.Insert(cluster);
          continue;
        }
      }

      // Check cluster coverage
      if (min_cluster_coverage > 0) {
        if (cluster->Coverage() < min_cluster_coverage) {
          nonviable_clusters.Insert(cluster);
          continue;
        }
      }

      // Check max_clusters
      if (max_clusters > 0) {
        if (viable_clusters.NEntries() >= max_clusters) {
          nonviable_clusters.Insert(cluster);
          continue;
        }
      }
    }
    
    // R3SegmentationCluster is viable
    cluster->segmentation = this;
    cluster->segmentation_index = viable_clusters.NEntries();    
    viable_clusters.Insert(cluster);
  }

  // Delete nonviable clusters
  for (int i = 0; i < nonviable_clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = nonviable_clusters.Kth(i);
    delete cluster;
  }

  // Replace clusters with viable ones
  clusters = viable_clusters;

  // Return success
  return 1;
}



int R3Segmentation::
MergeClusters(void)
{
  // Initialize statistics
  int cluster_count = clusters.NEntries();
  int merge_count = 0;
  int push_count = 0;

  //////////

  // Create pairs between clusters with neighbor points
  RNArray<R3SegmentationPair *> pairs;
  for (int i = 0; i < clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster0 = clusters.Kth(i);

    // Sample points
    const int max_points = 64;
    int jstep = cluster0->points.NEntries() / max_points;
    if (jstep == 0) jstep = 1;
    for (int j = 0; j < cluster0->points.NEntries(); j += jstep) {
      R3SegmentationPoint *point0 = cluster0->points.Kth(j);

      // Check neighbors
      for (int k = 0; k < point0->neighbors.NEntries(); k++) {
        R3SegmentationPoint *point1 = point0->neighbors.Kth(k);
        if (point0 == point1) continue;
        R3SegmentationCluster *cluster1 = point1->cluster;
        if (!cluster1) continue;
        if (cluster0 == cluster1) continue;

        // Check if already have pair
        if (FindR3SegmentationPair(cluster0, cluster1)) continue;

        // Compute affinity
        RNScalar affinity = cluster0->Affinity(cluster1);
        if ((affinity < min_pair_affinity) && (cluster_count <= max_clusters) && (min_cluster_points == 0) && (min_cluster_area == 0)) continue;

        // Create pair
        R3SegmentationPair *pair = new R3SegmentationPair(cluster0, cluster1, affinity);
        if (!pair) continue;

        // Insert pair
        pairs.Insert(pair);
      }        
    }
  }

  // Check if there are any pairs
  if (pairs.IsEmpty()) return 1;

  //////////

  // Initialize heap
  R3SegmentationPair tmp;
  RNHeap<R3SegmentationPair *> heap(&tmp, &tmp.affinity, &tmp.heapentry, FALSE);
  for (int i = 0; i < pairs.NEntries(); i++) {
    R3SegmentationPair *pair = pairs.Kth(i);
    heap.Push(pair);
  }

  // Merge clusters hierarchically
  while (!heap.IsEmpty()) {
    // Get pair
    R3SegmentationPair *pair = heap.Pop();

    // Check if we are done
    if ((pair->affinity < min_pair_affinity) && (cluster_count <= max_clusters) && (min_cluster_points == 0) && (min_cluster_area == 0)) {
      break;
    }

    // Get clusters
    R3SegmentationCluster *cluster0 = pair->clusters[0];
    R3SegmentationCluster *cluster1 = pair->clusters[1];

    // Check if either cluster has already been merged
    if (cluster0->parent || cluster1->parent) {
      // Find ancestors
      R3SegmentationCluster *ancestor0 = cluster0;
      R3SegmentationCluster *ancestor1 = cluster1;
      while (ancestor0->parent) ancestor0 = ancestor0->parent;
      while (ancestor1->parent) ancestor1 = ancestor1->parent;
      if (ancestor0 != ancestor1) {
        if (!FindR3SegmentationPair(ancestor0, ancestor1)) {
          RNScalar affinity = ancestor0->Affinity(ancestor1);
          if ((cluster_count > max_clusters) || (min_cluster_points > 0) || (min_cluster_area > 0) || (affinity >= min_pair_affinity)) {
            // Create a pair between the ancestors
            R3SegmentationPair *pair = new R3SegmentationPair(ancestor0, ancestor1, affinity);
            heap.Push(pair);
            push_count++;
          }
        }
      }
    }
    else {
      // Check if should merge
      RNBoolean merge = TRUE;
      if ((min_pair_affinity > 0) && (pair->affinity < min_pair_affinity)) merge = FALSE;
      if ((min_cluster_area > 0) && (cluster0->area < min_cluster_area)) merge = TRUE;
      if ((min_cluster_area > 0) && (cluster1->area < min_cluster_area)) merge = TRUE;
      if ((min_cluster_points > 0) && (cluster0->points.NEntries() < min_cluster_points)) merge = TRUE;
      if ((min_cluster_points > 0) && (cluster1->points.NEntries() < min_cluster_points)) merge = TRUE;
      if ((max_clusters > 0) && (cluster_count > max_clusters)) merge = TRUE;
      if (!merge) { delete pair; continue; }

#if 0
      // Print message
      if (print_progress) {
        static unsigned long count = 0;
        if ((count++ % 1000) == 0) {
          printf("        %15.12f : %9d %9d : %15d %15d %15d %15d\n", pair->affinity, 
                 cluster0->points.NEntries(), cluster1->points.NEntries(), 
                 heap.NEntries(), merge_count, push_count, cluster_count);
        }
      }
#endif
      
#if 0
      // Create merged cluster
      R3SegmentationCluster *cluster = new R3SegmentationCluster(cluster0, cluster1);
      clusters.Insert(cluster);
      cluster_count--;
      merge_count++;
#else
      // Merge smaller cluster into bigger one
      R3SegmentationCluster *parent = (cluster0->points.NEntries() > cluster1->points.NEntries()) ? cluster0 : cluster1;
      R3SegmentationCluster *child = (cluster0->points.NEntries() > cluster1->points.NEntries()) ? cluster1 : cluster0;
      parent->InsertChild(child);
      cluster_count--;
      merge_count++;
#endif
    }

    // Delete pair
    delete pair;
  }

  // Remove merged clusters
  RNArray<R3SegmentationCluster *> merged_clusters;
  RNArray<R3SegmentationCluster *> all_clusters = clusters;
  clusters.Empty();
  for (int i = 0; i < all_clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = all_clusters.Kth(i);
    cluster->segmentation = this;
    cluster->segmentation_index = clusters.NEntries();    
    if (!cluster->parent) { clusters.Insert(cluster); continue; }
    cluster->parent->RemoveChild(cluster);
    merged_clusters.Insert(cluster);
  }

  // Delete merged clusters
  for (int i = 0; i < merged_clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = merged_clusters.Kth(i);
    delete cluster;
  }

  // Return success
  return 1;
}



int R3Segmentation::
MergeSmallClusters(void)
{
  // Find small clusters
  RNArray<R3SegmentationCluster *> small_clusters;
  for (int i = 0; i < clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = clusters.Kth(i);
    assert(!cluster->parent);
    if (cluster->points.NEntries() >= min_cluster_points) continue;
    small_clusters.Insert(cluster);
  }

  // Check small clusters
  if (small_clusters.NEntries() < 2) return 1;

  // Merge n-1 small clusters into first one
  R3SegmentationCluster *cluster0 = small_clusters.Kth(0);
  for (int i = 1; i < small_clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = small_clusters.Kth(i);
    if (cluster == cluster0) continue;
    cluster0->InsertChild(cluster);
    assert(cluster->parent == cluster0);
    assert(cluster0->points.NEntries() > 0);
    assert(cluster->points.NEntries() == 0);
  }

  // Rebuild list of clusters
  RNArray<R3SegmentationCluster *> all_clusters = clusters;
  clusters.Empty();
  for (int i = 0; i < all_clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = all_clusters.Kth(i);
    if (!cluster->parent) {
      assert((cluster == cluster0) || (cluster->points.NEntries() >= min_cluster_points));
      cluster->segmentation = this;
      cluster->segmentation_index = clusters.NEntries();    
      clusters.Insert(cluster);
    }
    else {
      assert(cluster->points.NEntries() == 0);
      cluster->parent->RemoveChild(cluster);
      delete cluster;
    }
  }

  // Return success
  return 1;
}



int R3Segmentation::
SplitClusters(void)
{
#if 0
  // Check min clusters spacing
  if (min_cluster_spacing <= 0) return 1;

  // Split connected components
  RNArray<R3SegmentationCluster *> tmp = clusters;
  clusters.Empty();
  for (int i = 0; i < tmp.NEntries(); i++) {
    R3SegmentationCluster *cluster = tmp.Kth(i);

    // Check cluster area
    if (cluster->area < min_cluster_area) continue;

    // Check cluster points
    if (cluster->points.NEntries() < min_cluster_points) continue;

    // Rasterize points into planar grid
    R3PlanarGrid grid(cluster->primitive.plane, cluster->primitive.bbox, min_cluster_spacing);
    for (int j = 0; j < cluster->points.NEntries(); j++) {
      R3SegmentationPoint *point = cluster->points.Kth(j);
      R3Point position = point->position;
      grid.RasterizeWorldR3SegmentationPoint(position.X(), position.Y(), position.Z(), 1.0);
    }

    // Compute connected components
    int max_components = grid.NEntries();
    int *components = new int [ max_components ];
    int ncomponents = grid.ConnectedComponents(RN_EPSILON, max_components, NULL, NULL, components);

    // Check connected components
    if (ncomponents == 1) {
      // One connected component - simply insert cluster
      cluster->segmentation = this;
      cluster->segmentation_index = clusters.NEntries();    
      clusters.Insert(cluster);
    }
    else {
      // Create cluster for each connnected component
      for (int j = 0; j < ncomponents; j++) {
        // Make array of points in component
        RNArray<R3SegmentationPoint *> component_points;
        for (int k = 0; k < cluster->points.NEntries(); k++) {
          R3SegmentationPoint *point = cluster->points.Kth(k);
          R3Point world_position = point->position;
          R2R3SegmentationPoint grid_position = grid.GridPosition(world_position);
          int ix = (int) (grid_position.X() + 0.5);
          int iy = (int) (grid_position.Y() + 0.5);
          int index; grid.Grid().IndicesToIndex(ix, iy, index);
          if (components[index] != j) continue;
          component_points.Insert(point);
        }

        // Check number of points
        if (component_points.NEntries() > min_cluster_points) {

          // Find centroid
          R3Point centroid = R3zero_point;
          for (int k = 0; k < component_points.NEntries(); k++) {
            R3SegmentationPoint *point = component_points.Kth(k);
            R3Point world_position = point->position;
            centroid += world_position;
          }
          centroid /= component_points.NEntries();

          // Find seed point
          R3SegmentationPoint *seed_point = NULL;
          RNLength min_dd = FLT_MAX;
          for (int k = 0; k < component_points.NEntries(); k++) {
            R3SegmentationPoint *point = component_points.Kth(k);
            R3Point world_position = point->position;
            RNLength dd = R3SquaredDistance(centroid, world_position);
            if (dd < min_dd) { seed_point = point; min_dd = dd; }
          }

          // Check seed point
          if (seed_point) {
            // Create cluster
            R3SegmentationCluster *c = new R3SegmentationCluster(seed_point, R3_SEGMENTATION_PLANE_PRIMITIVE_TYPE);
            c->possible_affinity = cluster->possible_affinity;

            // Insert points into cluster
            for (int k = 0; k < component_points.NEntries(); k++) {
              R3SegmentationPoint *point = component_points.Kth(k);
              c->InsertPoint(point);
            }

            // Update primitive
            c->UpdatePrimitive();

            // Update color
            c->UpdateColor();

            // Update timestamp
            c->UpdateTimestamp();

            // Update planar grid
            // c->UpdatePlanarGrid();

            // Insert cluster
            c->segmentation = this;
            c->segmentation_index = clusters.NEntries();    
            clusters.Insert(c);
          }
        }
      }

      // Delete the original cluster
      delete cluster;
    }

    // Delete components
    delete [] components;
  }
#endif

  // Return success
  return 1;
}



int R3Segmentation::
RefineBoundaries(void)
{
  // Check if should refine boundaries
  if (!refine_boundaries) return 1;

  // Iteratively refine boundaries
  for (int iter = 0; iter < 1000; iter++) {
    RNBoolean done = TRUE;
    for (int i0 = 0; i0 < points.NEntries(); i0++) {
      R3SegmentationPoint *point0 = points.Kth(i0);
      if (point0->neighbors.NEntries() < 3) continue;
      R3SegmentationCluster *cluster0 = point0->cluster;
      if (!cluster0) continue;
      int i1 = RNRandomScalar() * point0->neighbors.NEntries();
      R3SegmentationPoint *point1 = point0->neighbors.Kth(i1);
      R3SegmentationCluster *cluster1 = point1->cluster;
      if (!cluster1) continue;
      if (cluster0 == cluster1) continue;
      int count = 0;
      for (int i2 = 0; i2 < point0->neighbors.NEntries(); i2++) {
        R3SegmentationPoint *point2 = point0->neighbors.Kth(i2);
        R3SegmentationCluster *cluster2 = point2->cluster;
        if (!cluster2) continue;
        if (cluster2 == cluster1) count++;
      }
      if (count > point0->neighbors.NEntries()/2) {
        cluster0->RemovePoint(point0);
        cluster1->InsertPoint(point0, point0->cluster_affinity);
        done = FALSE;
      }
    }
    if (done) break;
  }

  // Return success
  return 1;
}
                 


                 
////////////////////////////////////////////////////////////////////////
// Top-level segmentation functions
////////////////////////////////////////////////////////////////////////

int R3Segmentation::
CreateClusters(int primitive_type)
{
  // Print debug message
  RNTime step_time;
  step_time.Read();
  if (print_progress) {
    printf("      SA %.3f %d\n", step_time.Elapsed(), points.NEntries());
    step_time.Read();
  }

  // Create clusters
  if (initialize_hierarchically) {
    if (!CreateSingletonClusters(primitive_type)) return 0;
    if (!MergeClusters()) return 0;
  }
  else {
    if (!CreateRegionGrowingClusters(primitive_type)) return 0;
  }

  // Check clusters
  if (clusters.IsEmpty()) return 0;

  // Print debug message
  if (print_progress) {
    printf("      SB %.3f %d %d %g\n", step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
    step_time.Read();
  }

  // Iteratively update clusters
  for (int i = 0; i < max_refinement_iterations; i++) {
    // Reassign points to clusters
    if (!ReassignClusters()) return 0;

    // Print debug message
    if (print_progress) {
      printf("      SC %d : %.3f %d %d %g\n", i, step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
      step_time.Read();
    }

    // Create clusters
    if (!allow_outlier_points) {
      if (!CreateRegionGrowingClusters(primitive_type)) return 0;
    }

    // Print debug message
    if (print_progress) {
      printf("      SD %d : %.3f %d %d %g\n", i, step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
      step_time.Read();
    }

    // Merge clusters
    if (!MergeClusters()) return 0;

    // Print debug message
    if (print_progress) {
      printf("      SE %d : %.3f %d %d %g\n", i, step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
      step_time.Read();
    }

    // Refine boundaries
    if (!RefineBoundaries()) return 0;

    // Print debug message
    if (print_progress) {
      printf("      SF %d : %.3f %d %d %g\n", i, step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
      step_time.Read();
    }
    
    // Delete clusters
    if (!DeleteClusters()) return 0;

    // Print debug message
    if (print_progress) {
      printf("      SG %d : %.3f %d %d %g\n", i, step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
      step_time.Read();
    }
  }

  // Split clusters
  // if (!SplitClusters()) return 0;

  // Print debug message
  if (print_progress) {
    printf("      SH %.3f %d %d %g\n", step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
    step_time.Read();
  }

  // Refine boundaries
  if (!RefineBoundaries()) return 0;

  // Print debug message
  if (print_progress) {
    printf("      SI %.3f %d %d %g\n", step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
    step_time.Read();
  }

  // Delete clusters
  if (!DeleteClusters()) return 0;

  // Print debug message
  if (print_progress) {
    printf("      SJ %.3f %d %d %g\n", step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
    step_time.Read();
  }

  // Sort clusters
  clusters.Sort(CompareClusters);

  // Merge small clusters
  if (!MergeSmallClusters()) return 0;

  // Print debug message
  if (print_progress) {
    printf("      SK %.3f %d %d %g\n", step_time.Elapsed(), clusters.NEntries(), NUnclusteredPoints(), Affinity());
    step_time.Read();
  }
  // Return success
  return 1;
}



int R3Segmentation::
WriteFile(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    RNFail("Unable to open segmentation file %s\n", filename);
    return 0;
  }

  // Write clusters to file
  for (int i = 0; i < clusters.NEntries(); i++) {
    R3SegmentationCluster *cluster = clusters.Kth(i);
    R3Point center;
    RNScalar variances[3];
    R3Triad axes = cluster->PrincipleAxes(&center, variances);
    R3Box extent = cluster->AxisExtents();
    fprintf(fp, "%d %d %g %g %g %d  %g %g %g   %g %g %g %g   %g %g %g  %g  %g %g %g   %g %g %g  %g %g %g  %g %g %g   %g %g %g   %g %g %g  %g %g %g\n",
            i+1, cluster->points.NEntries(), cluster->area,
            cluster->total_affinity, cluster->possible_affinity, cluster->primitive.primitive_type,
            cluster->primitive.centroid.X(), cluster->primitive.centroid.Y(), cluster->primitive.centroid.Z(),
            cluster->primitive.plane.A(), cluster->primitive.plane.B(), cluster->primitive.plane.C(), cluster->primitive.plane.D(),
            cluster->color.R(), cluster->color.G(), cluster->color.B(),
            cluster->timestamp,
            center.X(), center.Y(), center.Z(),
            axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2], axes[2][0], axes[2][1], axes[2][2],
            variances[0], variances[1], variances[2],
            extent[0][0], extent[0][1], extent[0][2], extent[1][0], extent[1][1], extent[1][2]);
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R3Segmentation::
PrintParameters(FILE *fp) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print all parameters
  fprintf(fp, "Min clusters = %d\n", min_clusters);
  fprintf(fp, "Max clusters = %d\n", max_clusters);
  fprintf(fp, "Min cluster points = %d\n", min_cluster_points);
  fprintf(fp, "Min cluster area = %g\n", min_cluster_area);
  fprintf(fp, "Min cluster coverage = %g\n", min_cluster_coverage);
  fprintf(fp, "Max cluster diameter = %g\n", max_cluster_diameter);
  fprintf(fp, "Max cluster primitive distance = %g\n", max_cluster_primitive_distance);
  fprintf(fp, "Max cluster normal angle = %g\n", max_cluster_normal_angle);
  fprintf(fp, "Max cluster color difference = %g\n", max_cluster_color_difference);
  fprintf(fp, "Max cluster timestamp difference = %g\n", max_cluster_timestamp_difference);
  fprintf(fp, "Max cluster category difference = %g\n", max_cluster_category_difference);
  fprintf(fp, "Max pair centroid distance = %g\n", max_pair_centroid_distance);
  fprintf(fp, "Max pair primitive distance = %g\n", max_pair_primitive_distance);
  fprintf(fp, "Max pair normal angle = %g\n", max_pair_normal_angle);
  fprintf(fp, "Max pair color difference = %g\n", max_pair_color_difference);
  fprintf(fp, "Max pair timestamp difference = %g\n", max_pair_timestamp_difference);
  fprintf(fp, "Max pair category difference = %g\n", max_pair_category_difference);
  fprintf(fp, "Min pair affinity = %g\n", min_pair_affinity);
  fprintf(fp, "Max refinement iterations = %d\n", max_refinement_iterations);  
  fprintf(fp, "Max reassignment iterations = %d\n", max_reassignment_iterations);  
  fprintf(fp, "Equalize cluster sizes = %d\n", equalize_cluster_sizes);
  fprintf(fp, "Balance cluster sizes = %d\n", balance_cluster_sizes);
  fprintf(fp, "Favor convex clusters = %d\n", favor_convex_clusters);
  fprintf(fp, "Scale tolerances with depth = %d\n", scale_tolerances_with_depth);
  fprintf(fp, "Initialize hierarchically = %d\n", initialize_hierarchically);
  fprintf(fp, "Allow outlier points = %d\n", allow_outlier_points);
  fprintf(fp, "Refine boundaries = %d\n", refine_boundaries);

  // Return success
  return 1;
}
} // end namespace gaps
