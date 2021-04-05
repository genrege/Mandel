# Mandel
Mandelbrot/Julia and related fractal interactive rendering.

Basic instructions:
- Select the set to render from the dropdown
- Some sets have an additional coordinate - specify using Ctrl+Click in ther overview or main panels
- Alternate "Special Experimental" sets can be selected using the up/down selector control
- Mouse wheel zooms in/oput around the cursor
- Left click pans the image
- Some sets appear to work better with a lower number of iterations eg Julia - this may be a palette limitation, to be addressed
- For large numbers of iterations, see the Window TDR item before for configuration

Limitations/TODO:
- Uses Microsoft C++ AMP for GPU compute.  This only works in Release mode and appears to be a Microsoft bug.
- To be ported to OpenCL, work in progress
- Numeric limits of FP64 - fixed/arbitrary precision needed for deeper zooms
- C# UI improvements needed, especially some interactive palette selection - these are hardcoded right now
- Too much code in mandelbrotset.h, needs to be refactored (eg palettes)
- Windows TDR will crash out the GPU after some time doing calculations (default is only 2s)
  - only fix I've found is to edit the registry TdrDelay and TdrDdiDelay values to some large number (eg 1200 seconds)
  - these keys should be added as DWORD in Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers + "TdrDelay" and "TdrDdiDelay".  I have "TdrLevel"=3

Future functionality:
- generate video from interaction
- track + zoom plotting recording
- the above will allow fire and forget tracking and zooming to specific points
- rotate the coordinate system for funkier videos
- add 3D rendering/fly-throughs of the sets
- palette editing (possibly an external editing program with import into the render engine)

Hardware/OS:
- GPU: tested on AMD Radeon VII, NVidia 2070 Super and NVidia 1660 Super
- CPU: AMD 3900X, AMD 3600X
- Windows 10

