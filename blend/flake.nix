{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        config = {
          allowUnfree = true;
        };
        inherit system;
      };

      dist = import ./dist {inherit pkgs;};

      initScript = pkgs.writeScript "run.sh" ''
        # echo populate .gitignore
        # echo -e "\ndist/*\nflake.*\nconfig/*" > .gitignore

        # echo Reinstalling blender plugins ...

        ${builtins.foldl' (a: b: a + b.ri) "" dist.plugins}

        echo Starting tts server ...
        python dist/btts.py &

        bash

        trap '
          echo "Exiting the shell ... "
          rm -r config/Code/Workspaces/*
          pkill -f btts.py
        ' EXIT
      '';

      fhs = pkgs.buildFHSEnv rec {
        multiPkgs = pkgs.steam.args.multiPkgs; # for bforartists , universal
        name = "qnal-zone";
        # export PATH=$PATH:${blender.outPath}/bin
        profile = ''
          export XDG_CONFIG_HOME=$PWD/config
          export PATH=$PATH:${dist.bforartists}/
          export BLENDER_SYSTEM_PYTHON=${dist.bforartists_python}
        '';

        targetPkgs = pkgs:
          with pkgs; [
            ffmpeg
            # blender

            libdecor # for bforartists

            # wl-clipboard-x11

            (python3.withPackages (p:
              with p; [
                waitress
                flask
                numpy
                scipy

                # ipython
                debugpy

                requests
                filelock
                tqdm
                pyyaml
                pytaglib
                torchaudio
                omegaconf
                rich
                soundfile
                piper-phonemize
                tabulate
                # piper-tts
                # deepspeed

                (buildPythonPackage rec {
                  pname = "deepspeed";
                  version = "";
                  src = fetchGit {
                    url = "https://github.com/microsoft/DeepSpeed";
                    rev = "4d866bd55a6b2b924987603b599c1f8f35911c4b";
                  };
                  doCheck = false;
                })
                py-cpuinfo
                psutil
                hjson
                pydantic
                librosa
                pandas
                matplotlib

                # resemble-enhance
                (buildPythonPackage rec {
                  pname = "resemble-enhance";
                  version = "0.0.1";
                  src = fetchGit {
                    url = "https://github.com/resemble-ai/resemble-enhance";
                    rev = "b4bd8d693a8353617bba7d0cd0fb2e8c4e586527";
                  };
                  doCheck = false;
                })

                dtw-python
                openai-whisper
                (buildPythonPackage rec {
                  pname = "whisper-timestamped";
                  version = "";
                  src = fetchGit {
                    url = "https://github.com/linto-ai/whisper-timestamped";
                    rev = "a82e4d884a504625e8d6a98265272e2cb14c0901";
                  };
                  doCheck = false;
                })

                #
                huggingface-hub

                (buildPythonPackage rec {
                  pname = "fake-bpy-module-latest";
                  version = "20231106";
                  src = fetchPypi {
                    inherit pname version;
                    sha256 = "sha256-rq5XfPI1qSa+viHTqt2G+f/QiwAReay9t/StG9GTguE=";
                  };
                  doCheck = false;
                })

                # ansible
                # jmespath

                # MELOTTS
                torch
                torchaudio
                librosa
                pypinyin
                jieba
                transformers
                mecab-python3
                unidic-lite
                num2words
                pykakasi
                fugashi
                nltk
                inflect
                anyascii
                jamo
                gruut
                google-api-core
                google
                google-cloud-storage
                boto3
                rich
                (buildPythonPackage rec {
                  pname = "MeloTTS";
                  version = "0.0.1";
                  src = fetchGit {
                    url = "https://github.com/myshell-ai/MeloTTS.git";
                    rev = "5b538481e24e0d578955be32a95d88fcbde26dc8";
                  };
                  doCheck = false;
                })
                (buildPythonPackage rec {
                  pname = "g2p-en";
                  version = "2.1.0";
                  src = fetchGit {
                    url = "https://github.com/Kyubyong/g2p.git";
                    rev = "c6439c274c42b9724a7fee1dc07ca6a4c68a0538";
                  };
                  doCheck = false;
                })
                (buildPythonPackage rec {
                  pname = "proces";
                  version = "0.1.7";
                  src = fetchGit {
                    url = "https://github.com/Ailln/proces.git";
                    rev = "622d37aa378cdae2010ee40834067e4698f6ec3a";
                  };
                  doCheck = false;
                })
                (buildPythonPackage rec {
                  pname = "cn2an";
                  version = "0.5.22";
                  src = fetchGit {
                    url = "https://github.com/Ailln/cn2an.git";
                    rev = "40b6e3f53815be00af0392d40755c24401c7c61c";
                  };
                  doCheck = false;
                })
                # https://files.pythonhosted.org/packages/1c/3d/3e04a822b8615904269f7126d8b019ae5c3b5c3c78397ec8bab056b02099/cn2an-0.5.22-py3-none-any.whl
                # proces
                (buildPythonPackage rec {
                  pname = "cached-path";
                  version = "1.6.2";
                  src = fetchurl {
                    url = "https://files.pythonhosted.org/packages/eb/7b/b793dccfceb3d0de9e3f376f1a8e3a1e4015361ced5f96bb78c279d552be/cached_path-1.6.2-py3-none-any.whl";
                    sha256 = "sha256-Y85+aeTsjJ+1dzFKxTCYs8y+zO0llqepIdrVOXb/blo=";
                  };
                  format = "wheel";
                  doCheck = false;
                  buildInputs = [];
                  checkInputs = [];
                  nativeBuildInputs = [];
                  propagatedBuildInputs = [];
                })
              ]))
            dist.bforartists_python

            (vscode-with-extensions.override {
              # vscode = vscodium;
              vscodeExtensions = with vscode-extensions;
                [vscodevim.vim ms-python.python ms-vscode.cpptools]
                ++ pkgs.vscode-utils.extensionsFromVscodeMarketplace [
                  {
                    name = "blender-development";
                    publisher = "JacquesLucke";
                    version = "0.0.20";
                    sha256 = "sha256-UQzTwPZyElzwtWAjbkHIsun+VEttlph4Og6A6nFxk8w=";
                  }
                ];
            })

            (texlive.combine {
              inherit
                (pkgs.texlive)
                scheme-basic
                standalone
                preview # definately needed
                dvisvgm
                dvipng # for preview and export as html
                wrapfig
                amsmath
                ulem
                hyperref
                capt-of
                ; # probably needed
              #(setq org-latex-compiler "lualatex")
              #(setq org-preview-latex-default-process 'dvisvgm)
            })
          ];
        runScript = initScript;
      };
    in {
      devShells = {
        default = fhs.env;
      };
    });
}