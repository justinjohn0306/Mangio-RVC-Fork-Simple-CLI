import argparse
import sys
import torch
import json
from multiprocessing import cpu_count

global usefp16
usefp16 = False


def use_fp32_config():
    usefp16 = False
    device_capability = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Assuming you have only one GPU (index 0).
        device_capability = torch.cuda.get_device_capability(device)[0]
        if device_capability >= 7:
            usefp16 = True
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                with open(f"configs/{config_file}", "r") as d:
                    data = json.load(d)

                if "train" in data and "fp16_run" in data["train"]:
                    data["train"]["fp16_run"] = True

                with open(f"configs/{config_file}", "w") as d:
                    json.dump(data, d, indent=4)

                print(f"Set fp16_run to true in {config_file}")

            with open(
                "trainset_preprocess_pipeline_print.py", "r", encoding="utf-8"
            ) as f:
                strr = f.read()

            strr = strr.replace("3.0", "3.7")

            with open(
                "trainset_preprocess_pipeline_print.py", "w", encoding="utf-8"
            ) as f:
                f.write(strr)
        else:
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                with open(f"configs/{config_file}", "r") as f:
                    data = json.load(f)

                if "train" in data and "fp16_run" in data["train"]:
                    data["train"]["fp16_run"] = False

                with open(f"configs/{config_file}", "w") as d:
                    json.dump(data, d, indent=4)

                print(f"Set fp16_run to false in {config_file}")

            with open(
                "trainset_preprocess_pipeline_print.py", "r", encoding="utf-8"
            ) as f:
                strr = f.read()

            strr = strr.replace("3.7", "3.0")

            with open(
                "trainset_preprocess_pipeline_print.py", "w", encoding="utf-8"
            ) as f:
                f.write(strr)
    else:
        print(
            "CUDA is not available. Make sure you have an NVIDIA GPU and CUDA installed."
        )
    return (usefp16, device_capability)


class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.paperspace,
            self.is_cli,
            self.simple_cli,
            self.simple_cli_args,
        ) = self.arg_parse()
        
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        subparser = parser.add_subparsers()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(  # Fork Feature. Paperspace integration for web UI
            "--paperspace",
            action="store_true",
            help="Note that this argument just shares a gradio link for the web UI. Thus can be used on other non-local CLI systems.",
        )
        parser.add_argument(  # Fork Feature. Embed a CLI into the infer-web.py
            "--is_cli",
            action="store_true",
            help="Use the CLI instead of setting up a gradio UI. This flag will launch an RVC text interface where you can execute functions from infer-web.py!",
        )
        parser.add_argument( # Fork Feature. Embed a CLI into the infer-web.py
            "--simple_cli", choices=["infer", "pre-process", "extract-feature", "train", "train-feature", "extract-model", "uvr", ""], default="", help="Use the simpler CLI instead of the cli interface. Choose from 1) pre-process 2) extract-feature 3)  WIP."
        )

        # Arguments for simple cli usage.
        parser.add_argument(
            "--exp_name", type=str, default="mi-test", help="Experiment name"
        )
        parser.add_argument(
            "--trainset_dir",
            type=str,
            default="",
            help="Trainset directory",
        )
        parser.add_argument(
            "--sample_rate", choices=["32k", "40k", "48k"], default="40k", help="Sample rate: 40k (32k, 40k, 48k)"
        )
        parser.add_argument(
            "--n_workers", type=int, default=8, help="Number of cpu threads to work"
        )
        parser.add_argument(
            "--gpu", type=int, default=0, help="GPU device index to use"
        )
        parser.add_argument(
            "--is_pitch_guidance",
            type=bool,
            default=True,
            help="Use pitch guidance (1 for True 0 for False)",
        )
        parser.add_argument(
            "--f0_method",
            type=str,
            default="crepe",
            help="F0 extraction method",
        )
        parser.add_argument(
            "--crepe_hop_length",
            type=int,
            default=128,
            help="Hop length for crepe",
        )
        parser.add_argument(
            "--rvc_version",
            choices=["v1", "v2"],
            default="v2",
            help="RVC version",
        )
        parser.add_argument(
            "--speaker_id",
            type=int,
            default=0,
            help="Speaker id for multi-speaker model",
        )
        parser.add_argument(
            "--save_epoch_iter",
            type=int,
            default=5,
            help="Save model every n iterations",
        )
        parser.add_argument(
            "--epochs", type=int, default=20, help="Number of epochs to train"
        )
        parser.add_argument(
            "--batch_size", type=int, default=8, help="Batch size for training"
        )
        parser.add_argument(
            "--latest_ckpt_only",
            type=bool,
            default=False,
            help="Save only the latest checkpoint",
        )
        parser.add_argument(
            "--cache_trainset",
            type=bool,
            default=False,
            help="Whether to cache training set to vram",
        )
        parser.add_argument(
            "--save_small_model",
            type=bool,
            default=False,
            help="Save extracted small model every generation?",
        )

        parser.add_argument(
            "--model_file_name",
            type=str,
            default="",
            help="Model name with .pth in ./weights",
        )
        parser.add_argument(
            "--source_audio_path",
            type=str,
            default="",
            help="Source audio path for inference",
        )
        parser.add_argument(
            "--output_file_name",
            type=str,
            default="output.wav",
            help="Output file name to be placed in './audio-outputs'",
        )
        parser.add_argument(
            "--feature_index_path",
            type=str,
            default="",
            help="Feature index file path",
        )
        parser.add_argument(
            "--transposition",
            type=int,
            default=0,
            help="Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)",
        )
        parser.add_argument(
            "--infer_f0_method",
            choices=["pm", "harvest", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"],
            default="crepe",
            help="F0 extraction method for inference",
        )
        parser.add_argument(
            "--harvest_median_filter_radius",
            type=int,
            default=3,
            help="Harvest median filter radius, default 3.",
        )
        parser.add_argument(
            "--post_sample_rate",
            type=int,
            default=0,
            help="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling.",
        )
        parser.add_argument(
            "--mix_volume_envelope",
            type=float,
            default=0.25,
            help="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used.",
        )
        parser.add_argument(
            "--feature_index_ratio",
            type=float,
            default=0.33,
            help="Feature index ratio for inference.",
        )
        parser.add_argument(
            "--voiceless_consonant_protection",
            type=float,
            default=0.33,
            help="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy."
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="",
            help="Model path for extract-model",
        )
        parser.add_argument(
            "--model_save_name",
            type=str,
            default="",
            help="Model save name for extract-model",
        )
        parser.add_argument(
            "--model_info",
            type=str,
            default="",
            help="Model info for extract-model",
        )
        parser.add_argument(
            "--cmd_help",
            action="store_true",
            help="Print help for simple cli",
        )
        # Add --agg and --format
        parser.add_argument(
            "--agg",
            type=int,
            default=10,
            help="Aggregation for uvr5",
        )
        parser.add_argument(
            "--format",
            type=str,
            default="flac",
            help="Audio format",
        )
        parser.add_argument(
            "--uvr5_weight_name",
            type=str,
            default="",
            help="UVR5 weight name",
        )
        parser.add_argument(
            "--formant_shift",
            action="store_true",
            help="Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)",
        )
        parser.add_argument(
            "--formant_quefrency",
            type=float,
            default=8.0,
            help="Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)",
        )
        parser.add_argument(
            "--formant_timbre",
            type=float,
            default=1.2,
            help="Timbre for formanting: 1.2 (no need to set if arg14 is False/false)",
        )

        cmd_opts = parser.parse_args()

        args_to_assign = ['exp_name', 'trainset_dir', 'sample_rate', 'n_workers', 'gpu',
                  'is_pitch_guidance', 'f0_method', 'crepe_hop_length', 'rvc_version',
                  'speaker_id', 'save_epoch_iter', 'epochs', 'batch_size',
                  'latest_ckpt_only', 'cache_trainset', 'save_small_model',
                  'model_file_name', 'source_audio_path', 'output_file_name',
                  'feature_index_path', 'transposition', 'infer_f0_method',
                  'harvest_median_filter_radius', 'post_sample_rate',
                  'mix_volume_envelope', 'feature_index_ratio',
                  'voiceless_consonant_protection', 'model_path', 
                  'model_save_name', 'model_info', 'cmd_help', 'agg', 'format', 'uvr5_weight_name',
                  'formant_shift', 'formant_quefrency', 'formant_timbre']
        simple_cli_args = argparse.Namespace(**{arg: getattr(cmd_opts, arg) for arg in args_to_assign})

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.paperspace,
            cmd_opts.is_cli,
            cmd_opts.simple_cli,
            simple_cli_args,
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("Found GPU", self.gpu_name, ", force to fp32")
                self.is_half = False
            else:
                print("Found GPU", self.gpu_name)
                use_fp32_config()
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif self.has_mps():
            print("No supported Nvidia GPU found, use MPS instead")
            self.device = "mps"
            self.is_half = False
            use_fp32_config()
        else:
            print("No supported Nvidia GPU found, use CPU instead")
            self.device = "cpu"
            self.is_half = False
            use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
