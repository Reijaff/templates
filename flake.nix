{
  description = "Ready-made templates for easily creating flake-driven environments";

  outputs = {self}: {
    templates = {
      go = {
        path = ./go;
      };
      cpp-hello = {
        path = ./cpp-hello;
      };
      cpp-mingw = {
        path = ./cpp-mingw;
      };
      julia = {
        path = ./julia;
      };
      python = {
        path = ./python;
      };
      zig = {
        path = ./zig;
      };
      blend = {
        path = ./blend;
      };
      rust = {
        path = ./rust;
      };
    };
  };
}
