# SoMapping
## clang-format
command clang-format for folder
```bash
find plane_fusion_final -iname *.h -o -iname *.cpp | xargs clang-format -i -style=Google
```
```bash
find plane_fusion_final -iname *.cuh -o -iname *.cu | xargs clang-format -i -style=Google
```
