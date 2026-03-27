{
  description = "RF signal IQ dataset generator for ML training";

  # Canonical URL: github:Quantum-Serendipity/rf-datagen
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs, ... }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (system:
        f (import nixpkgs { inherit system; })
      );

      # Shared definitions keyed by pkgs, evaluated once per system.
      rfLib = pkgs: let
        python3 = pkgs.python3;
      in {
        pysstv = python3.pkgs.buildPythonPackage rec {
          pname = "pysstv";
          version = "0.5.7";
          pyproject = true;
          src = pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-iQahNQDLmyVY9VPtahdzNhXJXt12bYSRPuZKSlkhhJU=";
          };
          build-system = [ python3.pkgs.setuptools ];
          dependencies = [ python3.pkgs.pillow ];
          doCheck = false;
        };

        runtimeCliTools = [
          pkgs.fldigi pkgs.wsjtx pkgs.direwolf
          pkgs.codec2 pkgs.m17-cxx-demod
          pkgs.piper-tts pkgs.espeak-ng
          pkgs.pulseaudio pkgs.xvfb-run
          # Validation decoders
          pkgs.dsdcc pkgs.multimon-ng
          pkgs.whisper-cpp-vulkan
        ];
      };
    in {
      packages = forAllSystems (pkgs: let
        python3 = pkgs.python3;
        lib = rfLib pkgs;
      in {
        default = python3.pkgs.buildPythonApplication {
          pname = "rf-datagen";
          version = "1.0.0";
          pyproject = true;
          src = ./.;
          build-system = [ python3.pkgs.setuptools ];
          dependencies = [
            python3.pkgs.numpy
            python3.pkgs.scipy
            python3.pkgs.pillow
            lib.pysstv
          ];
          makeWrapperArgs = [
            "--prefix" "PATH" ":" (pkgs.lib.makeBinPath (lib.runtimeCliTools ++ [ pkgs.coreutils ]))
            "--set" "ALSA_PLUGIN_DIR" "${pkgs.alsa-plugins}/lib/alsa-lib"
          ];
          doCheck = false;
          meta.mainProgram = "rf-datagen";
        };

        # Library package for import by other flakes (no CLI tool wrappers)
        pythonPackage = python3.pkgs.buildPythonPackage {
          pname = "rf-datagen";
          version = "1.0.0";
          pyproject = true;
          src = ./.;
          build-system = [ python3.pkgs.setuptools ];
          dependencies = [
            python3.pkgs.numpy
            python3.pkgs.scipy
            python3.pkgs.pillow
            lib.pysstv
          ];
          doCheck = false;
        };
      });

      devShells = forAllSystems (pkgs: let
        python3 = pkgs.python3;
        lib = rfLib pkgs;

        devPython = python3.withPackages (ps: [
          ps.numpy ps.scipy ps.pillow ps.matplotlib
          ps.pytest
          lib.pysstv
        ]);
      in {
        default = pkgs.mkShellNoCC {
          # Only devPython goes in packages — CLI tools are added to
          # PATH explicitly below to prevent piper-tts (a Python pkg)
          # from propagating its entire dep tree into PYTHONPATH via
          # nix setup hooks.
          packages = [ devPython ];
          shellHook = ''
            export PATH="${pkgs.lib.makeBinPath lib.runtimeCliTools}:$PATH"
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            export ALSA_PLUGIN_DIR=${pkgs.alsa-plugins}/lib/alsa-lib
          '';
        };
      });
    };
}
