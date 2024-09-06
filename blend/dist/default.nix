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
      pillow
      tqdm

      (buildPythonPackage rec {
        pname = "scenedetect";
        version = "";
        src = fetchGit {
          url = "https://github.com/Breakthrough/PySceneDetect";
          rev = "53d2441d060117d440857f260710fc6d7ad67c2a";
        };
        doCheck = false;
      })

      # rembg ( crashes )
      # pooch
      # jsonschema
      # pymatting
      # # opencv4
      # onnxruntime
      # (buildPythonPackage rec {
      #   pname = "rembg";
      #   version = "";
      #   src = fetchGit {
      #     url = "https://github.com/danielgatis/rembg";
      #     rev = "95b81143c9a1d760c892ffa7f406f055fc244b81";
      #   };
      #   doCheck = false;
      # })
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
      repo = "freesound";
      owner = "iwkse";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "7ac73f675187934f03b964b0b3e63e0a6951abc1";
      };
      ri = runtimeInstallScript src repo;
    }

    rec {
      repo = "abratools";
      owner = "abrasic";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "df705c0c6ba5a9a327b1dcf5bf884ba545b6463d";
      };
      ri = runtimeInstallScript src repo;
    }

    rec {
      repo = "import_latex_as_curve";
      owner = "Reijaff";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "c699829e6da3983acb7366bd200a9550fbeef60a";
      };
      ri = runtimeInstallScript src repo;
    }

    rec {
      repo = "anim_utils";
      owner = "Reijaff";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "7277c669ab3ad1846827397c27b2e6d38e2f575b";
      };
      ri = runtimeInstallScript src repo;
    }

    rec {
      repo = "remove_background";
      owner = "Reijaff";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "f360a55da3b05cd724c1a31cd3eb531d9e3f5ddb";
      };
      ri = runtimeInstallScript src repo;
    }

    rec {
      repo = "Shot_Detection";
      owner = "Reijaff";
      src = builtins.fetchGit {
        url = "https://github.com/${owner}/${repo}";
        rev = "c5e3ead25e6f7e275135cb1aa2731b76f80ebb15";
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