#ifndef _SLAB_GEOM_HEADER_
#define _SLAB_GEOM_HEADER_

class SlabGeometry {
  public:
    std::vector<std::shared_ptr<Material>> mat;
    std::vector<double>   xcoords;
    uint32_t              ints;

    size_t size() { return mat.size(); }
};

#endif
