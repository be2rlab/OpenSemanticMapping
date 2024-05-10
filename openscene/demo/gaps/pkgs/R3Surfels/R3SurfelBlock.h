/* Include file for the R3 surfel block class */
#ifndef __R3__SURFEL__BLOCK__H__
#define __R3__SURFEL__BLOCK__H__



////////////////////////////////////////////////////////////////////////
// NAMESPACE 
////////////////////////////////////////////////////////////////////////

namespace gaps {



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelBlock {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelBlock(void);
  R3SurfelBlock(int nsurfels);
  R3SurfelBlock(const R3SurfelBlock& block);
  R3SurfelBlock(const R3SurfelPointSet *set);
  R3SurfelBlock(const R3SurfelPointSet *set,
    const R3Point& position_origin, RNScalar timestamp_origin = 0);
  R3SurfelBlock(const R3Surfel *surfels, int nsurfels,
    const R3Point& position_origin = R3zero_point, RNScalar timestamp_origin = 0);
  R3SurfelBlock(const RNArray<const R3Surfel *>& array,
    const R3Point& position_origin = R3zero_point, RNScalar timestamp_origin = 0);
  R3SurfelBlock(const R3Point *points, int npoints);
  R3SurfelBlock(const RNArray<R3Point *>& points);

  // Destructor function
  ~R3SurfelBlock(void);


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Node access functions
  R3SurfelNode *Node(void) const;

  // Database access functions
  R3SurfelDatabase *Database(void) const;
  int DatabaseIndex(void) const;

  // Surfel access functions
  int NSurfels(void) const;
  const R3Surfel *Surfels(void) const;
  const R3Surfel *Surfel(int k) const;
  const R3Surfel *operator[](int k) const;


  //////////////////////////////////
  //// BLOCK PROPERTY FUNCTIONS ////
  //////////////////////////////////

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;
  RNScalar Resolution(void) const;
  RNLength AverageRadius(void) const;
  const R3Point& PositionOrigin(void) const;

  // Elevation property functions
  RNInterval ElevationRange(void) const;
  
  // Timestamp property functions
  const RNInterval& TimestampRange(void) const;
  RNScalar TimestampOrigin(void) const;

  // Identifier property functions
  unsigned int MinIdentifier(void) const;
  unsigned int MaxIdentifier(void) const;
  
  // Aggregate surfel property functions
  RNBoolean HasActive(void) const;
  RNBoolean HasNormals(void) const;
  RNBoolean HasTangents(void) const;
  RNBoolean HasAerial(void) const;
  RNBoolean HasTerrestrial(void) const;

  // User data property functions
  void *Data(void) const;

  
  ///////////////////////////////////
  //// SURFEL PROPERTY FUNCTIONS ////
  ///////////////////////////////////

  // Surfel property functions
  int SurfelIndex(const R3Surfel *surfel) const;
  R3Point SurfelPosition(int surfel_index) const;
  R3Vector SurfelNormal(int surfel_index) const;
  R3Vector SurfelTangent(int surfel_index) const;
  RNLength SurfelRadius(int surfel_index) const;
  RNLength SurfelRadius(int surfel_index, int axis) const;
  RNLength SurfelDepth(int surfel_index) const;
  RNLength SurfelElevation(int surfel_index) const;
  RNRgb SurfelColor(int surfel_index) const;
  RNScalar SurfelTimestamp(int surfel_index) const;
  unsigned int SurfelIdentifier(int surfel_index) const;
  unsigned int SurfelAttribute(int surfel_index) const;
  RNBoolean IsSurfelActive(int surfel_index) const;
  RNBoolean IsSurfelMarked(int surfel_index) const;
  RNBoolean IsSurfelAerial(int surfel_index) const;
  RNBoolean IsSurfelTerrestrial(int surfel_index) const;
  RNBoolean IsSurfelOriented(int surfel_index) const;
  RNBoolean IsSurfelIsotropic(int surfel_index) const;
  RNBoolean IsSurfelOnSilhouetteBoundary(int surfel_index) const;
  RNBoolean IsSurfelOnShadowBoundary(int surfel_index) const;
  RNBoolean IsSurfelOnBorderBoundary(int surfel_index) const;
  RNBoolean IsSurfelOnBoundary(int surfel_index) const;


  //////////////////////////////////////////
  //// BLOCK MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Assignment operator
  R3SurfelBlock& operator=(const R3SurfelBlock& block);
  
  // Property manipulation functions
  void SetPositionOrigin(const R3Point& position);
  void SetTimestampOrigin(RNScalar timestamp);
  void SetMarks(RNBoolean mark = TRUE);
  void SetData(void *data);


  ///////////////////////////////////////
  //// SURFEL MANIPULATION FUNCTIONS ////
  ///////////////////////////////////////

  // Surfel manipulation functions
  void SetSurfelPosition(int surfel_index, const R3Point& position);
  void SetSurfelNormal(int surfel_index, const R3Vector& normal);
  void SetSurfelTangent(int surfel_index, const R3Vector& tangent);
  void SetSurfelRadius(int surfel_index, RNLength radius);
  void SetSurfelRadius(int surfel_index, int axis, RNLength radius);
  void SetSurfelDepth(int surfel_index, RNLength depth);
  void SetSurfelElevation(int surfel_index, RNLength elevation);
  void SetSurfelColor(int surfel_index, const RNRgb& color);
  void SetSurfelTimestamp(int surfel_index, RNLength timestamp);
  void SetSurfelIdentifier(int surfel_index, unsigned int identifier);
  void SetSurfelAttribute(int surfel_index, unsigned int attribute);
  void SetSurfelFlags(int surfel_index, unsigned char flags);
  void SetSurfelActive(int surfel_index, RNBoolean active = TRUE);
  void SetSurfelAerial(int surfel_index, RNBoolean aerial = TRUE);
  void SetSurfelSilhouetteBoundary(int surfel_index, RNBoolean boundary = TRUE);
  void SetSurfelShadowBoundary(int surfel_index, RNBoolean boundary = TRUE);
  void SetSurfelBorderBoundary(int surfel_index, RNBoolean boundary = TRUE);
  void SetSurfelMark(int surfel_index, RNBoolean mark = TRUE);

  // Transformation functions
  void Transform(const R3Affine& transformation);


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Draw function
  void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS, int subsampling_factor = 1) const;

  // Print function
  void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;

  //////////////////////////////////////////
  //// I/O functions
  //////////////////////////////////////////

  // File I/O
  int ReadFile(const char *filename);
  int ReadXYZAsciiFile(const char *filename);
  int ReadXYZBinaryFile(const char *filename);
  int ReadBinaryFile(const char *filename);
  int ReadUPCFile(const char *filename);
  int ReadOBJFile(const char *filename);
  int ReadXYZAscii(FILE *fp);
  int ReadXYZBinary(FILE *fp);
  int ReadBinary(FILE *fp);
  int ReadUPC(FILE *fp);
  int ReadOBJ(FILE *fp);
  int WriteFile(const char *filename) const;
  int WriteXYZAsciiFile(const char *filename) const;
  int WriteBinaryFile(const char *filename) const;
  int WriteXYZAscii(FILE *fp) const;
  int WriteBinary(FILE *fp) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

  // Identifier manipulation functions
  void SetMinIdentifier(unsigned int identifier);
  void SetMaxIdentifier(unsigned int identifier);

  // For backward compatibility
  const R3Point& Origin(void) const;
  void SetOrigin(const R3Point& origin);

  // Manipulation functions
  RNBoolean IsDirty(void) const;
  void SetDirty(RNBoolean dirty = TRUE);
  
public:
  // Update functions
  void UpdateProperties(void);

  // Only use if you know what you are doing
  int ReadCount(void) const { return file_read_count; }

  // Only use if you know what you are doing (really)
  void ResetSurfels(int nsurfels);
  
protected:
  // Database update functions
  void UpdateAfterInsert(R3SurfelDatabase *database);
  void UpdateBeforeRemove(R3SurfelDatabase *database);

  // Tree node update functions
  void UpdateAfterInsert(R3SurfelNode *node);
  void UpdateBeforeRemove(R3SurfelNode *node);

private:
  // Manipulation functions
  void SetDatabase(R3SurfelDatabase *database);

  // Block update functions
  void UpdateBBox(void);
  void UpdateTimestampRange(void);
  void UpdateIdentifierRange(void);
  void UpdateResolution(void);
  void UpdateFlags(void);

  // Surfel update functions
  void UpdateSurfelNormals(void);

private:
  // Surfel data
  R3Surfel *surfels;
  int nsurfels;

  // Property data
  R3Point position_origin;
  R3Box bbox;
  RNScalar timestamp_origin;
  RNInterval timestamp_range;
  unsigned int min_identifier;
  unsigned int max_identifier;
  RNScalar resolution;
  RNFlags flags;
  void *data;

  // Database data
  friend class R3SurfelDatabase;
  friend class R3SurfelPointSet;
  class R3SurfelDatabase *database;
  int database_index;
  unsigned long long file_surfels_offset;
  unsigned int file_surfels_count;
  unsigned int file_read_count;

  // Node data
  friend class R3SurfelNode;
  R3SurfelNode *node;

  // Display data
  GLuint opengl_id;
};



////////////////////////////////////////////////////////////////////////
// BLOCK FLAGS
////////////////////////////////////////////////////////////////////////

#define R3_SURFEL_BLOCK_PROPERTY_FLAGS                 0x00FF
#define R3_SURFEL_BLOCK_BBOX_UPTODATE_FLAG             0x0001
#define R3_SURFEL_BLOCK_RESOLUTION_UPTODATE_FLAG       0x0002
#define R3_SURFEL_BLOCK_FLAGS_UPTODATE_FLAG            0x0004

#define R3_SURFEL_BLOCK_HAS_AERIAL_FLAG                0x0010
#define R3_SURFEL_BLOCK_HAS_TERRESTRIAL_FLAG           0x0020
#define R3_SURFEL_BLOCK_HAS_ACTIVE_FLAG                0x0040
#define R3_SURFEL_BLOCK_HAS_NORMALS_FLAG               0x0080
#define R3_SURFEL_BLOCK_HAS_TANGENTS_FLAG              0x0008

#define R3_SURFEL_BLOCK_DATABASE_FLAGS                 0xFF00
#define R3_SURFEL_BLOCK_DIRTY_FLAG                     0x0100
#define R3_SURFEL_BLOCK_DELETE_PENDING_FLAG            0x0200



////////////////////////////////////////////////////////////////////////
// Drawing method
////////////////////////////////////////////////////////////////////////

// Only one should be defined
#define R3_SURFEL_BLOCK_DRAW_WITH_GLBEGIN              0
#define R3_SURFEL_BLOCK_DRAW_WITH_DISPLAY_LIST         1
#define R3_SURFEL_BLOCK_DRAW_WITH_VBO                  2
#define R3_SURFEL_BLOCK_DRAW_WITH_ARRAYS               3
#define R3_SURFEL_BLOCK_DRAW_METHOD R3_SURFEL_BLOCK_DRAW_WITH_GLBEGIN



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline const R3Point& R3SurfelBlock::
PositionOrigin(void) const
{
  // Return position of origin
  // Surfels coordinates are relative to this position
  return position_origin;
}



inline RNScalar R3SurfelBlock::
TimestampOrigin(void) const
{
  // Return timestamp origin
  // Surfel timestamps are relative to this time
  return timestamp_origin;
}



inline R3Point R3SurfelBlock::
Centroid(void) const
{
  // Return centroid of bounding box
  return BBox().Centroid();
}



inline R3SurfelDatabase *R3SurfelBlock::
Database(void) const
{
  // Return database
  return database;
}



inline int R3SurfelBlock::
DatabaseIndex(void) const
{
  // Return index of block in database
  return database_index;
}



inline R3SurfelNode *R3SurfelBlock::
Node(void) const
{
  // Return tree node this block is part of
  return node;
}



inline int R3SurfelBlock::
NSurfels(void) const
{
  // Return number of surfels
  return nsurfels;
}



inline const R3Surfel *R3SurfelBlock::
Surfels(void) const
{
  // Return pointer to block of surfels
  return surfels;
}



inline const R3Surfel *R3SurfelBlock::
Surfel(int k) const
{
  // Return kth surfel
  assert((k >= 0) && (k < nsurfels));
  return &surfels[k];
}



inline const R3Surfel *R3SurfelBlock::
operator[](int k) const
{
  // Return kth surfel
  return Surfel(k);
}



inline RNLength R3SurfelBlock::
AverageRadius(void) const
{
  // Return average radius of surfels
  RNScalar res = Resolution();
  if (res == 0) return 0;
  return sqrt(1.0 / (res * RN_PI));
}



inline void *R3SurfelBlock::
Data(void) const
{
  // Return user data
  return data;
}



inline int R3SurfelBlock::
SurfelIndex(const R3Surfel *surfel) const
{
  // Return index of surfel in this block
  if (!surfels) return 0;
  int index = surfel - surfels;
  assert((index >= 0) && (index < nsurfels));
  return index;
}



inline R3Point R3SurfelBlock::
SurfelPosition(int surfel_index) const
{
  // Return position of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return R3Point(position_origin.X() + surfel.PX(),
                 position_origin.Y() + surfel.PY(),
                 position_origin.Z() + surfel.PZ());
}



inline R3Vector R3SurfelBlock::
SurfelNormal(int surfel_index) const
{
  // Return normal of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return R3Vector(surfel.NX(), surfel.NY(), surfel.NZ());
}



inline R3Vector R3SurfelBlock::
SurfelTangent(int surfel_index) const
{
  // Return tangent of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return R3Vector(surfel.TX(), surfel.TY(), surfel.TZ());
}



inline RNLength R3SurfelBlock::
SurfelRadius(int surfel_index) const
{
  // Return radius of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return surfel.Radius();
}



inline RNLength R3SurfelBlock::
SurfelRadius(int surfel_index, int axis) const
{
  // Return radius of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return surfel.Radius(axis);
}



inline RNLength R3SurfelBlock::
SurfelDepth(int surfel_index) const
{
  // Return depth of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return surfel.Depth();
}



inline RNLength R3SurfelBlock::
SurfelElevation(int surfel_index) const
{
  // Return elevation of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return surfel.Elevation();
}



inline RNRgb R3SurfelBlock::
SurfelColor(int surfel_index) const
{
  // Return color of kth surfel
  return surfels[surfel_index].Rgb();
}



inline RNScalar R3SurfelBlock::
SurfelTimestamp(int surfel_index) const
{
  // Return timestamp of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return timestamp_origin + surfel.Timestamp();
}



inline unsigned int R3SurfelBlock::
SurfelIdentifier(int surfel_index) const
{
  // Return identifier of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return surfel.Identifier();
}



inline unsigned int R3SurfelBlock::
SurfelAttribute(int surfel_index) const
{
  // Return attribute of kth surfel
  R3Surfel& surfel = surfels[surfel_index];
  return surfel.Attribute();
}



inline RNBoolean R3SurfelBlock::
IsSurfelActive(int surfel_index) const
{
  // Return whether kth surfel is active
  return surfels[surfel_index].IsActive();
}



inline RNBoolean R3SurfelBlock::
IsSurfelMarked(int surfel_index) const
{
  // Return whether kth surfel is marked
  return surfels[surfel_index].IsMarked();
}



inline RNBoolean R3SurfelBlock::
IsSurfelOriented(int surfel_index) const
{
  // Return whether kth surfel is oriented 
  return surfels[surfel_index].IsOriented();
}



inline RNBoolean R3SurfelBlock::
IsSurfelIsotropic(int surfel_index) const
{
  // Return whether kth surfel is isotropic
  return surfels[surfel_index].IsIsotropic();
}



inline RNBoolean R3SurfelBlock::
IsSurfelAerial(int surfel_index) const
{
  // Return whether kth surfel is isotropic
  return surfels[surfel_index].IsAerial();
}



inline RNBoolean R3SurfelBlock::
IsSurfelTerrestrial(int surfel_index) const
{
  // Return whether kth surfel is aerial
  return surfels[surfel_index].IsTerrestrial();
}



inline RNBoolean R3SurfelBlock::
IsSurfelOnSilhouetteBoundary(int surfel_index) const
{
  // Return whether kth surfel is on silhouette boundary
  return surfels[surfel_index].IsOnSilhouetteBoundary();
}



inline RNBoolean R3SurfelBlock::
IsSurfelOnShadowBoundary(int surfel_index) const
{
  // Return whether kth surfel is on shadow boundary
  return surfels[surfel_index].IsOnShadowBoundary();
}



inline RNBoolean R3SurfelBlock::
IsSurfelOnBorderBoundary(int surfel_index) const
{
  // Return whether kth surfel is on border boundary
  return surfels[surfel_index].IsOnBorderBoundary();
}



inline RNBoolean R3SurfelBlock::
IsSurfelOnBoundary(int surfel_index) const
{
  // Return whether kth surfel is on  boundary
  return surfels[surfel_index].IsOnBoundary();
}



inline const R3Point& R3SurfelBlock::
Origin(void) const
{
  // Return position of position origin
  // DO NOT USE -- here only for backward compatibility
  return PositionOrigin();
}



inline void R3SurfelBlock::
SetOrigin(const R3Point& origin)
{
  // Set position origin
  // DO NOT USE -- here only for backward compatibility
  SetPositionOrigin(origin);
}



inline void R3SurfelBlock::
SetMinIdentifier(unsigned int identifier)
{
  // Set min identifier
  this->min_identifier = identifier;
}



inline void R3SurfelBlock::
SetMaxIdentifier(unsigned int identifier)
{
  // Set max identifier
  this->max_identifier = identifier;
}



inline void R3SurfelBlock::
UpdateAfterInsert(R3SurfelDatabase *database)
{
}



inline void R3SurfelBlock::
UpdateBeforeRemove(R3SurfelDatabase *database)
{
}



// End namespace
}


// End include guard
#endif
