{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    platform = "mingwW64";
    pkgs = nixpkgs.legacyPackages.${system};
    mingw_pkgs = nixpkgs.legacyPackages.${system}.pkgsCross.${platform};
  in {
    devShells.${system}.default = mingw_pkgs.mkShell {
      buildInputs = with pkgs; [
        cmake
        gnumake
        ninja
      ];
      shellHook = ''
        cat >.clangd <<EOF
        CompileFlags:
          Compiler: ${mingw_pkgs.buildPackages.gcc.cc}/bin/x86_64-w64-mingw32-gcc
          Add: [-I${mingw_pkgs.buildPackages.gcc.cc}/x86_64-w64-mingw32/sys-include, --target=x86_64-w64-mingw32]
        EOF


        $EDITOR ./
      '';
    };
  };
}
