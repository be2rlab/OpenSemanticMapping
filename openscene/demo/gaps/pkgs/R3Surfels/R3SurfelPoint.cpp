/* Source file for the R3 surfel point class */



/* Include files */

#include "R3Surfels.h"



// Namespace

namespace gaps {



/* Public functions */

R3SurfelPoint::
R3SurfelPoint(void)
  : block(NULL),
    surfel(NULL)
{
}



R3SurfelPoint::
R3SurfelPoint(const R3SurfelPoint& point)
  : block(point.block),
    surfel(point.surfel)
{
  // Update block reference count
  if (block && block->Database()) block->Database()->ReadBlock(block);  
}



R3SurfelPoint::
R3SurfelPoint(R3SurfelBlock *block, const R3Surfel *surfel)
  : block(block),
    surfel(surfel)
{
  // Update block reference count
  if (block && block->Database()) block->Database()->ReadBlock(block);  
}



R3SurfelPoint::
R3SurfelPoint(R3SurfelBlock *block, int surfel_index)
  : block(block),
    surfel(NULL)
{
  // Update block reference count
  if (block && block->Database()) block->Database()->ReadBlock(block);

  // Find surfel
  surfel = block->Surfel(surfel_index);
}



R3SurfelPoint::
~R3SurfelPoint(void)
{
  // Update block reference count
  if (block && block->Database()) block->Database()->ReleaseBlock(block);  
}



void R3SurfelPoint::
Copy(const R3SurfelPoint *point)
{
  // Update block reference counts
  if (this->block != point->block) {
    if (this->block && this->block->Database()) this->block->Database()->ReleaseBlock(this->block);
    if (point->block && point->block->Database()) point->block->Database()->ReadBlock(point->block);
  }

  // Copy block and surfel
  this->block = point->block;
  this->surfel = point->surfel;
}



void R3SurfelPoint::
Reset(R3SurfelBlock *block, const R3Surfel *surfel)
{
  // Update block reference counts
  if (this->block != block) {
    if (this->block && this->block->Database()) this->block->Database()->ReleaseBlock(this->block);
    if (block && block->Database()) block->Database()->ReadBlock(block);
  }

  // Copy block and surfel
  this->block = block;
  this->surfel = surfel;
}



R3SurfelPoint& R3SurfelPoint::
operator=(const R3SurfelPoint& point)
{
  // Copy the point
  Copy(&point);
  return *this;
}



void R3SurfelPoint::
SetPosition(const R3Point& position)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelPosition(block->SurfelIndex(surfel), position);
}



void R3SurfelPoint::
SetNormal(const R3Vector& normal)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelNormal(block->SurfelIndex(surfel), normal);
}



void R3SurfelPoint::
SetTangent(const R3Vector& tangent)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelTangent(block->SurfelIndex(surfel), tangent);
}



void R3SurfelPoint::
SetRadius(float radius)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelRadius(block->SurfelIndex(surfel), radius);
}



void R3SurfelPoint::
SetRadius(int axis, float radius)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelRadius(block->SurfelIndex(surfel), axis, radius);
}



void R3SurfelPoint::
SetColor(const RNRgb& color)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelColor(block->SurfelIndex(surfel), color);
}



void R3SurfelPoint::
SetDepth(RNLength depth)
{
  // Set depth
  assert(block && surfel);
  block->SetSurfelDepth(block->SurfelIndex(surfel), depth);
}



void R3SurfelPoint::
SetElevation(RNLength elevation)
{
  // Set elevation
  assert(block && surfel);
  block->SetSurfelElevation(block->SurfelIndex(surfel), elevation);
}



void R3SurfelPoint::
SetTimestamp(RNScalar timestamp)
{
  // Set timestamp
  assert(block && surfel);
  block->SetSurfelTimestamp(block->SurfelIndex(surfel), timestamp);
}



void R3SurfelPoint::
SetIdentifier(unsigned int identifier)
{
  // Set identifier
  assert(block && surfel);
  block->SetSurfelIdentifier(block->SurfelIndex(surfel), identifier);
}



void R3SurfelPoint::
SetAttribute(unsigned int attribute)
{
  // Set attribute
  assert(block && surfel);
  block->SetSurfelAttribute(block->SurfelIndex(surfel), attribute);
}



void R3SurfelPoint::
SetActive(RNBoolean active)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelActive(block->SurfelIndex(surfel), active);
}



void R3SurfelPoint::
SetAerial(RNBoolean aerial)
{
  // Set position
  assert(block && surfel);
  block->SetSurfelAerial(block->SurfelIndex(surfel), aerial);
}



void R3SurfelPoint::
SetMark(RNBoolean mark)
{
  // Set mark 
  // Should this be persistent???  No for now
  R3Surfel *s = (R3Surfel *) this->surfel;
  s->SetMark(mark);
}



void R3SurfelPoint::
Draw(RNFlags flags) const
{
  // Draw surfel
  RNGrfxBegin(RN_GRFX_POINTS);
  if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) RNLoadRgb(Color());
  R3LoadPoint(Position().Coords());
  RNGrfxEnd();
}



R3Point
SurfelPointPosition(R3SurfelPoint *point, void *)
{
  return point->Position();
}



} // namespace gaps
