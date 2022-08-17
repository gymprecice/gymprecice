// ********************  geometry  ********************
radius = 0.05;
xmax = 2;
xmin = -0.2;
ymax = .21;
ymin = -0.2;
x_coarse_line = 0.5;
mesh_thickness = 0.01;
cylinder_msh_res = 0.005;
inlet_msh_res = 0.01;
outlet_mesh_res = 0.02;

// ********************  points  ********************
//  circle
Point(1) = {0, 0, 0, cylinder_msh_res};
Point(2) = {0, radius, 0, cylinder_msh_res};
Point(3) = {-radius, 0, 0, cylinder_msh_res};
Point(4) = {0, -radius, 0, cylinder_msh_res};
Point(5) = {radius, 0, 0, cylinder_msh_res};

// fine-region points
Point(6) = {xmin, ymin, 0, inlet_msh_res};
Point(7) = {x_coarse_line, ymin, 0, outlet_mesh_res};
Point(8) = {x_coarse_line, ymax, 0, outlet_mesh_res};
Point(9) = {xmin, ymax, 0, inlet_msh_res};

// coarse-region points
Point(10) = {xmax, ymin, 0, outlet_mesh_res};
Point(11) = {xmax, ymax, 0, outlet_mesh_res};

// ********************  lines  ********************
// circle
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

// fine-mesh bounding box
Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 9};
Line(8) = {9, 6};

// coarse-mesh bounding box
Line(9) = {7, 10};
Line(10) = {10, 11};
Line(11) = {11, 8};

// ********************  surfaces  ********************
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Curve Loop(3) = {-6, 9, 10, 11};

Plane Surface(1) = {2, 1};
Plane Surface(2) = {3};

// ********************  make 3D  ********************
surfaceVector[] = Extrude {0, 0, mesh_thickness} {
     Surface{1,2};
     Layers{1};
     Recombine;
};

// ********************  physical boundaries  ********************
Physical Surface("back") = {1,2};
Physical Surface("front") = {surfaceVector[0],surfaceVector[10]};
Physical Surface("bottom") = {surfaceVector[2], surfaceVector[13]};
Physical Surface("inlet") = surfaceVector[5];
Physical Surface("top") = {surfaceVector[4], surfaceVector[15]};
Physical Surface("cylinder_wall") = {surfaceVector[6], surfaceVector[7], surfaceVector[8], surfaceVector[9]};
Physical Surface("outlet") = surfaceVector[14];

// ********************  physical Volume  ********************
Physical Volume("internal") = {surfaceVector[1], surfaceVector[11]};