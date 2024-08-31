{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      # packages.default = llvmPackages.stdenv.mkDerivation rec {
      # pname = "myapp";
      # version = "0.1.0";

      # src = ./.;

      # nativeBuildInputs = [cmake];
      # buildInputs = [];

      # cmakeFlags = [
      # "-DENABLE_TESTING=OFF"
      # "-DENABLE_INSTALL=ON"
      # ];
      # };

      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          cmake
          gnumake
          ninja

          pkgs.hello
        ];

        shellHook = ''
          $EDITOR ./
        '';
      };
    });
}
