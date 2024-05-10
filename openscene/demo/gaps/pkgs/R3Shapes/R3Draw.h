/* Include file for R3 draw utility */
#ifndef __R3__DRAW__H__
#define __R3__DRAW__H__



/* Begin namespace */
namespace gaps {



/* Inline macros */

#define R3LoadRgb RNLoadRgb



/* Inline functions */

inline void 
R3LoadNormal(const R3Vector& normal)
{
    // Load normal vector
    R3LoadNormal(normal.Coords());
}



inline void 
R3LoadPoint(const R3Point& point)
{
    // Load vertex 
    R3LoadPoint(point.Coords());
}



inline void 
R3LoadTextureCoords(const R2Point& texcoords)
{
    // Set texture coordinate (within R3BeginXXX and R3EndXXX)
    R3LoadTextureCoords(texcoords.Coords());
}



inline void 
R3DrawText(const R3Point& p, const char *str, void *font = NULL)
{
    // Draw text at point
    R3DrawText(p.X(), p.Y(), p.Z(), str, font);
}



// End namespace
}


// End include guard
#endif
