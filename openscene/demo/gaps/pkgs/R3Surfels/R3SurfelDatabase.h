/* Include file for the R3 surfel database class */
#ifndef __R3__SURFEL__DATABASE__H__
#define __R3__SURFEL__DATABASE__H__



////////////////////////////////////////////////////////////////////////
// NAMESPACE 
////////////////////////////////////////////////////////////////////////

namespace gaps {



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelDatabase {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelDatabase(void);
  R3SurfelDatabase(const R3SurfelDatabase& database);

  // Destructor function
  virtual ~R3SurfelDatabase(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Property functions
  const char *Name(void) const;
  long long NSurfels(void) const;
  unsigned int MajorVersion(void) const;
  unsigned int MinorVersion(void) const;

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;

  // Timestamp property functions
  const RNInterval& TimestampRange(void) const;

  // Identifier property functions
  unsigned int MaxIdentifier(void) const;
  

  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Tree access functions
  R3SurfelTree *Tree(void) const;

  // Block access functions
  int NBlocks(void) const;
  R3SurfelBlock *Block(int k) const;


  //////////////////////////////////////////
  //// PROPERTY MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Property manipulation functions
  void SetName(const char *name);


  //////////////////////////////////////////
  //// STRUCTURE MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Block manipulation functions
  virtual void InsertBlock(R3SurfelBlock *block);
  virtual void RemoveBlock(R3SurfelBlock *block);
  virtual void RemoveAndDeleteBlock(R3SurfelBlock *block);

  // High-level block manipulation functions
  virtual int InsertSubsetBlocks(R3SurfelBlock *block, 
    const RNArray<const R3Surfel *>& subsetA, const RNArray<const R3Surfel *>& subsetB, 
    R3SurfelBlock **blockA, R3SurfelBlock **blockB);


  ///////////////////////////////////////
  //// SURFEL MANIPULATION FUNCTIONS ////
  ///////////////////////////////////////

  // Surfel manipulation functions
  void SetMarks(RNBoolean mark = TRUE);


  /////////////////////////////////////
  //// MEMORY MANAGEMENT FUNCTIONS ////
  /////////////////////////////////////

  // Memory management functions
  int ReadBlock(R3SurfelBlock *block);
  int ReleaseBlock(R3SurfelBlock *block);
  int SyncBlock(R3SurfelBlock *block);
  RNBoolean IsBlockResident(R3SurfelBlock *block) const;
  unsigned long ResidentSurfels(void) const;


  ///////////////////////
  //// I/O FUNCTIONS ////
  ///////////////////////

  // I/O functions for out-of-core manipulation
  virtual int OpenFile(const char *filename, const char *rwaccess = NULL);
  virtual int SyncFile(void);
  virtual int CloseFile(void);
  virtual RNBoolean IsOpen(void) const;

  // I/O functions for files
  virtual int ReadFile(const char *filename);
  virtual int WriteFile(const char *filename);

  // I/O functions for streams
  virtual int ReadStream(FILE *fp);
  virtual int WriteStream(FILE *fp);


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Draw function
  virtual void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

public:
  // Filename query functions
  const char *Filename(void) const;

  // Identifier manipulation functions
  void SetMaxIdentifier(unsigned int identifier);

  // Internal block manipulation functions
  virtual int PurgeDeletedBlocks(void);

  // Internal surfel size functions
  int NBytesPerSurfel(void) const;

protected:
  // Internal block I/O functions
  virtual int InternalReadBlock(R3SurfelBlock *block, FILE *fp, int swap_endian);
  virtual int InternalReleaseBlock(R3SurfelBlock *block, FILE *fp, int swap_endian);
  virtual int InternalSyncBlock(R3SurfelBlock *block, FILE *fp, int swap_endian);

  // Internal surfel I/O functions
  virtual int ReadSurfel(FILE *fp, R3Surfel *ptr, int count, int swap_endian, 
    unsigned int major_version, unsigned int minor_version) const;
  virtual int WriteSurfel(FILE *fp, R3Surfel *ptr, int count, int swap_endian, 
    unsigned int major_version, unsigned int minor_version) const;

  // Internal header I/O functions
  virtual int ReadFileHeader(FILE *fp, unsigned int& nblocks);
  virtual int ReadBlockHeader(FILE *fp, unsigned int nblocks, int swap_endian);
  virtual int WriteFileHeader(FILE *fp, int swap_endian);
  virtual int WriteBlockHeader(FILE *fp, int swap_endian);

protected:
  FILE *fp;
  char *filename;
  char *rwaccess;
  unsigned int major_version;
  unsigned int minor_version;
  unsigned int swap_endian;
  unsigned long long file_blocks_offset;
  unsigned int file_blocks_count;
  RNArray<R3SurfelBlock *> blocks;
  long long nsurfels;
  R3Box bbox;
  RNInterval timestamp_range;
  unsigned int max_identifier;
  char *name;
  friend class R3SurfelTree;
  R3SurfelTree *tree;
  unsigned long resident_surfels;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline const char *R3SurfelDatabase::
Name(void) const
{
  // Return name
  return name;
}



inline long long R3SurfelDatabase::
NSurfels(void) const
{
  // Return total number of surfels in all blocks
  return nsurfels;
}



inline unsigned int R3SurfelDatabase::
MajorVersion(void) const
{
  // Return major version
  return major_version;
}



inline unsigned int R3SurfelDatabase::
MinorVersion(void) const
{
  // Return minor version
  return minor_version;
}



inline const R3Box& R3SurfelDatabase::
BBox(void) const
{
  // Return bounding box of database
  return bbox;
}



inline R3Point R3SurfelDatabase::
Centroid(void) const
{
  // Return centroid of database
  return BBox().Centroid();
}



inline const RNInterval& R3SurfelDatabase::
TimestampRange(void) const
{
  // Return timestamp range of database
  return timestamp_range;
}



inline unsigned int R3SurfelDatabase::
MaxIdentifier(void) const
{
  // Return maximum identifier in database
  return max_identifier;
}



inline R3SurfelTree *R3SurfelDatabase::
Tree(void) const
{
  // Return tree
  return tree;
}



inline int R3SurfelDatabase::
NBlocks(void) const
{
  // Return number of blocks
  return blocks.NEntries();
}



inline R3SurfelBlock *R3SurfelDatabase::
Block(int k) const
{
  // Return kth block
  return blocks[k];
}


inline RNBoolean R3SurfelDatabase::
IsOpen(void) const
{
  // Return whether database is open
  return (fp) ? TRUE : FALSE;
}



inline RNBoolean R3SurfelDatabase::
IsBlockResident(R3SurfelBlock *block) const
{
  // Return whether block is resident in memory
  return (block->surfels) ? TRUE : FALSE;
}



inline unsigned long R3SurfelDatabase::
ResidentSurfels(void) const
{
  // Return number of resident surfels
  return resident_surfels;
}



inline void R3SurfelDatabase::
SetMaxIdentifier(unsigned int identifier)
{
  // Set max identifier
  this->max_identifier = identifier;
}



inline int R3SurfelDatabase::
ReadBlock(R3SurfelBlock *block)
{
  // Check whether block needs to be read
  if (block->file_read_count == 0) {
    if (!InternalReadBlock(block, fp, swap_endian)) return 0;
  }

  // Increment reference count
  block->file_read_count++;

  // Return success
  return 1;
}



inline int R3SurfelDatabase::
ReleaseBlock(R3SurfelBlock *block)
{
  // Check whether block needs to be written
  if (block->file_read_count == 1) {
    if (!InternalReleaseBlock(block, fp, swap_endian)) return 0;
  }

  // Decrement reference count
  block->file_read_count--;

  // Check if delete pending
  if (block->file_read_count == 0) {
    if (block->flags[R3_SURFEL_BLOCK_DELETE_PENDING_FLAG]) {
      RemoveBlock(block);
      delete block;
    }
  }

  // Return success
  return 1;
}



inline int R3SurfelDatabase::
SyncBlock(R3SurfelBlock *block)
{
  // Check whether block needs to be written
  if (block->IsDirty()) {
    if (!InternalSyncBlock(block, fp, swap_endian)) return 0;
    block->SetDirty(FALSE);
  }

  // Return success
  return 1;
}



inline const char *R3SurfelDatabase::
Filename(void) const
{
  // Return scene filename
  return filename;
}

  
  
// End namespace
}


// End include guard
#endif
