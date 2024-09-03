{pkgs, ...}: rec {
  bforartists = builtins.fetchTarball {
    url = "https://github.com/Bforartists/Bforartists/releases/download/v4.2.2/Bforartists-4.2.2-Linux.tar.xz";
    sha256 = "0camh86s4v5iv1dybnlszd7n4qjgs2mjwadw7xl5lawdb1y8b66i";
  };

  bforartists_python = pkgs.python311.withPackages (p:
    with p; [
      debugpy
      flask
      requests

      numpy
      opencv4
      tqdm
    ]);

  runtimeInstallScript = src: name: ''
    echo Installing ${name}

    addon_path=$XDG_CONFIG_HOME/bforartists/4.3/scripts/addons/${name}/
    rm -rf $addon_path
    mkdir -p $addon_path
    cp -r ${src}/* $addon_path
    chmod 755 -R $addon_path
  '';

  plugins = [
    rec {
      repo = "import_latex_as_curve";
      owner = "Reijaff";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "c699829e6da3983acb7366bd200a9550fbeef60a";
      };
      ri = runtimeInstallScript src repo;
    }

    # rec {
    #   repo = "marking_of_highlights";
    #   owner = "Reijaff";
    #   src = builtins.fetchGit {
    #     url = "https://github.com/${owner}/${repo}";
    #     rev = "7e0e2b229c9886fe3f5beafb04cbf0f284e75ab5";
    #   };
    #   ri = runtimeInstallScript src repo;
    # }

    # rec {
    #   repo = "bake_audio_frequencies";
    #   owner = "Reijaff";
    #   src = builtins.fetchGit {
    #     url = "https://github.com/${owner}/${repo}";
    #     rev = "5cb875600a0533c517307421df58a8c7883e7f75";
    #   };
    #   ri = runtimeInstallScript src repo;
    # }

    rec {
      repo = "plane_quad_mask";
      owner = "Reijaff";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "7bcea50f0b4ba785636a2bfa2a5902068f5beeba";
      };
      ri = runtimeInstallScript src repo;
    }

    rec {
      repo = "tts_client";
      owner = "Reijaff";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "46f6c54540dc54a18e04e91925113ce58a363f3d";
      };
      ri = runtimeInstallScript src repo;
    }
  ];
}