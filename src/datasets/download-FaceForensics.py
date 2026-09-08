#!/usr/bin/env python
"""Downloads FaceForensics++ public data release."""

# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

from os.path import join
from tqdm import tqdm


FILELIST_URL = "misc/filelist.json"

DEEPFAKE_DETECTION_URL = (
    "misc/deepfake_detection_filenames.json"
)

DEEPFAKES_MODEL_NAMES = [
    "decoder_A.h5",
    "decoder_B.h5",
    "encoder.h5",
]


DATASETS = {
    "original_youtube_videos":
        "misc/downloaded_youtube_videos.zip",

    "original_youtube_videos_info":
        "misc/downloaded_youtube_videos_info.zip",

    "original":
        "original_sequences/youtube",

    "DeepFakeDetection_original":
        "original_sequences/actors",

    "Deepfakes":
        "manipulated_sequences/Deepfakes",

    "DeepFakeDetection":
        "manipulated_sequences/DeepFakeDetection",

    "Face2Face":
        "manipulated_sequences/Face2Face",

    "FaceShifter":
        "manipulated_sequences/FaceShifter",

    "FaceSwap":
        "manipulated_sequences/FaceSwap",

    "NeuralTextures":
        "manipulated_sequences/NeuralTextures",
}


ALL_DATASETS = [
    "original",
    "DeepFakeDetection_original",
    "Deepfakes",
    "DeepFakeDetection",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]


COMPRESSION_LEVELS = [
    "raw",
    "c23",
    "c40",
]


FILE_TYPES = [
    "videos",
    "masks",
    "models",
]


SERVERS = [
    "EU",
    "EU2",
    "CA",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Downloads FaceForensics++ "
            "public data release."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Output directory.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASETS.keys()) + ["all"],
    )

    parser.add_argument(
        "-c",
        "--compression",
        type=str,
        default="raw",
        choices=COMPRESSION_LEVELS,
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="videos",
        choices=FILE_TYPES,
    )

    parser.add_argument(
        "-n",
        "--num_videos",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--server",
        type=str,
        default="EU",
        choices=SERVERS,
    )

    args = parser.parse_args()

    if args.server == "EU":

        server_url = (
            "http://canis.vc.in.tum.de:8100/"
        )

    elif args.server == "EU2":

        server_url = (
            "http://kaldir.vc.in.tum.de/"
            "faceforensics/"
        )

    elif args.server == "CA":

        server_url = (
            "http://falas.cmpt.sfu.ca:8100/"
        )

    else:

        raise ValueError(
            f"Unsupported server: {args.server}"
        )

    args.tos_url = (
        server_url
        + "webpage/FaceForensics_TOS.pdf"
    )

    args.base_url = (
        server_url + "v3/"
    )

    args.deepfakes_model_url = (
        server_url
        + "v3/manipulated_sequences/"
        + "Deepfakes/models/"
    )

    return args


def reporthook(
    count,
    block_size,
    total_size,
):

    global start_time

    if count == 0:

        start_time = time.time()

        return

    duration = max(
        time.time() - start_time,
        1e-6,
    )

    progress_size = int(
        count * block_size
    )

    speed = int(
        progress_size
        / (1024 * duration)
    )

    percent = 0

    if total_size > 0:

        percent = int(
            count
            * block_size
            * 100
            / total_size
        )

    sys.stdout.write(
        "\rProgress: "
        f"{percent}% | "
        f"{progress_size / (1024 * 1024):.1f} MB | "
        f"{speed} KB/s | "
        f"{duration:.0f}s"
    )

    sys.stdout.flush()


def download_file(
    url,
    output_file,
    report_progress=False,
):

    output_directory = os.path.dirname(
        output_file
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    if os.path.isfile(output_file):

        tqdm.write(
            "Skipping existing file: "
            + output_file
        )

        return

    file_handle, temporary_file = (
        tempfile.mkstemp(
            dir=output_directory
        )
    )

    os.close(file_handle)

    try:

        if report_progress:

            urllib.request.urlretrieve(
                url,
                temporary_file,
                reporthook=reporthook,
            )

        else:

            urllib.request.urlretrieve(
                url,
                temporary_file,
            )

        os.replace(
            temporary_file,
            output_file,
        )

    except Exception:

        if os.path.exists(
            temporary_file
        ):

            os.remove(
                temporary_file
            )

        raise


def download_files(
    filenames,
    base_url,
    output_path,
):

    os.makedirs(
        output_path,
        exist_ok=True,
    )

    for filename in tqdm(
        filenames
    ):

        download_file(
            base_url + filename,
            join(
                output_path,
                filename,
            ),
        )


def load_json(url):

    with urllib.request.urlopen(
        url
    ) as response:

        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def build_filelist(
    args,
    dataset_path,
):

    if (
        "DeepFakeDetection" in dataset_path
        or "actors" in dataset_path
    ):

        filepaths = load_json(
            args.base_url
            + "/"
            + DEEPFAKE_DETECTION_URL
        )

        if "actors" in dataset_path:

            filelist = filepaths[
                "actors"
            ]

        else:

            filelist = filepaths[
                "DeepFakesDetection"
            ]

    elif "original" in dataset_path:

        file_pairs = load_json(
            args.base_url
            + "/"
            + FILELIST_URL
        )

        filelist = []

        for pair in file_pairs:

            filelist.extend(
                pair
            )

    else:

        file_pairs = load_json(
            args.base_url
            + "/"
            + FILELIST_URL
        )

        filelist = []

        for pair in file_pairs:

            filelist.append(
                "_".join(pair)
            )

            if args.type != "models":

                filelist.append(
                    "_".join(
                        pair[::-1]
                    )
                )

    if (
        args.num_videos is not None
        and args.num_videos > 0
    ):

        print(
            f"Downloading first "
            f"{args.num_videos} files."
        )

        filelist = filelist[
            :args.num_videos
        ]

    return filelist


def main(args):

    print(
        "By continuing, you confirm "
        "agreement with the "
        "FaceForensics++ Terms of Use:"
    )

    print(args.tos_url)

    input(
        "Press Enter to continue: "
    )

    selected_datasets = (
        [args.dataset]
        if args.dataset != "all"
        else ALL_DATASETS
    )

    os.makedirs(
        args.output_path,
        exist_ok=True,
    )

    for dataset in selected_datasets:

        dataset_path = DATASETS[
            dataset
        ]

        print(
            f"Downloading {dataset}"
        )

        filelist = build_filelist(
            args,
            dataset_path,
        )

        if args.type == "videos":

            download_url = (
                args.base_url
                + f"{dataset_path}/"
                + f"{args.compression}/"
                + "videos/"
            )

            output_path = join(
                args.output_path,
                dataset_path,
                args.compression,
                "videos",
            )

            files = [
                filename + ".mp4"
                for filename in filelist
            ]

            print(
                "Output:",
                output_path,
            )

            download_files(
                files,
                download_url,
                output_path,
            )

        elif args.type == "masks":

            if (
                "original" in dataset
                or dataset == "FaceShifter"
            ):

                print(
                    "Masks unavailable for:",
                    dataset,
                )

                continue

            download_url = (
                args.base_url
                + f"{dataset_path}/"
                + "masks/videos/"
            )

            output_path = join(
                args.output_path,
                dataset_path,
                "masks",
                "videos",
            )

            files = [
                filename + ".mp4"
                for filename in filelist
            ]

            download_files(
                files,
                download_url,
                output_path,
            )

        elif args.type == "models":

            if dataset != "Deepfakes":

                print(
                    "Models only available "
                    "for Deepfakes."
                )

                continue

            for folder in tqdm(
                filelist
            ):

                folder_url = (
                    args.deepfakes_model_url
                    + folder
                    + "/"
                )

                folder_output = join(
                    args.output_path,
                    dataset_path,
                    "models",
                    folder,
                )

                for model_file in (
                    DEEPFAKES_MODEL_NAMES
                ):

                    download_file(
                        folder_url + model_file,
                        join(
                            folder_output,
                            model_file,
                        ),
                    )


if __name__ == "__main__":

    args = parse_args()

    main(args)
