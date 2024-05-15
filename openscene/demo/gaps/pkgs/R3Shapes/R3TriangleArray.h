/* Include file for the R3 triangle class */
#ifndef __R3__TRIANGLE__ARRAY__H__
#define __R3__TRIANGLE__ARRAY__H__



/* Begin namespace */
namespace gaps {



/* Initialization functions */

int R3InitTriangleArray();
void R3StopTriangleArray();



/* Triangle class definition */

class R3TriangleArray : public R3Surface {
    public:
        // Constructor functions
        R3TriangleArray(void);
        R3TriangleArray(const R3TriangleArray& array);
        R3TriangleArray(const RNArray<R3TriangleVertex *>& vertices, const RNArray<R3Triangle *>& triangles);
        virtual ~R3TriangleArray(void);
  
        // Triangle array properties
        const R3Box& Box(void) const;
        const RNFlags Flags(void) const;

	// Vertex access functions/operators
        int NVertices(void) const;
	R3TriangleVertex *Vertex(int index) const;

	// Triangle access functions/operators
        int NTriangles(void) const;
	R3Triangle *Triangle(int index) const;

        // Query functions/operations
        RNBoolean HasNormals(void) const;
        RNBoolean HasColors(void) const;
        RNBoolean HasTextureCoords(void) const;
  
        // Shape property functions/operators
	virtual const RNBoolean IsPoint(void) const;
	virtual const RNBoolean IsLinear(void) const;
	virtual const RNBoolean IsPlanar(void) const;
	virtual const RNBoolean IsConvex(void) const;
        virtual const RNInterval NFacets(void) const;
        virtual const RNLength Length(void) const;
        virtual const RNArea Area(void) const;
        virtual const R3Point Centroid(void) const;
        virtual const R3Point ClosestPoint(const R3Point& point) const;
        virtual const R3Point FurthestPoint(const R3Point& point) const;
        virtual const R3Shape& BShape(void) const;
        virtual const R3Box BBox(void) const;
        virtual const R3Sphere BSphere(void) const;

        // Manipulation functions/operators
	virtual void Flip(void);
	virtual void Mirror(const R3Plane& plane);
	virtual void Transform(const R3Transformation& transformation);
        virtual void Subdivide(RNLength max_edge_length);
        virtual void CreateVertexNormals(RNAngle max_angle = RN_PI/4.0);
	virtual void MoveVertex(R3TriangleVertex *vertex, const R3Point& position);
        virtual void LoadMesh(const R3Mesh& mesh);
	virtual void Update(void);
        R3TriangleArray& operator=(const R3TriangleArray& array);

        // Draw functions/operators
        virtual void Draw(const R3DrawFlags draw_flags = R3_DEFAULT_DRAW_FLAGS) const;

	// Standard shape definitions
	RN_CLASS_TYPE_DECLARATIONS(R3TriangleArray);
        R3_SHAPE_RELATIONSHIP_DECLARATIONS(R3TriangleArray);


     private:
        // Internal VBO drawing functions
        virtual void InvalidateVBO(void);
        virtual void UpdateVBO(void);
        virtual void DrawVBO(const R3DrawFlags draw_flags) const;

     private:
        // Triangle array data
        RNArray<R3TriangleVertex *> vertices;
	RNArray<R3Triangle *> triangles;
        R3Box bbox;
        RNFlags flags;

     private:
        // VBO drawing data
        unsigned int vbo_id;
        unsigned int vbo_size;
};



/* Inline functions */

inline const R3Box& R3TriangleArray::
Box(void) const
{
    // Return bounding box
    return bbox;
}



inline int R3TriangleArray::
NVertices(void) const
{
    // Return number of vertices
    return vertices.NEntries();
}



inline R3TriangleVertex *R3TriangleArray::
Vertex(int k) const
{
    // Return kth vertex
    return vertices[k];
}



inline int R3TriangleArray::
NTriangles(void) const
{
    // Return number of triangles
    return triangles.NEntries();
}



inline R3Triangle *R3TriangleArray::
Triangle(int k) const
{
    // Return kth triangle
    return triangles[k];
}



inline const RNFlags R3TriangleArray::
Flags(void) const
{
    // Return flags
    return flags;
}



inline RNBoolean R3TriangleArray::
HasNormals(void) const
{
  // Return whether all triangle vertices have explicit normals
  return flags[R3_VERTEX_NORMALS_DRAW_FLAG];
}



inline RNBoolean R3TriangleArray::
HasColors(void) const
{
  // Return whether all triangle vertices have explicit colors
  return flags[R3_VERTEX_COLORS_DRAW_FLAG];
}



inline RNBoolean R3TriangleArray::
HasTextureCoords(void) const
{
  // Return whether all triangle vertices have explicit texture coordinates
  return flags[R3_VERTEX_TEXTURE_COORDS_DRAW_FLAG];
}



// End namespace
}


// End include guard
#endif
