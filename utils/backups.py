import os
import shutil
import hashlib
import time

speaker_name = input("Enter the speaker name: ")  # User input for speaker name
SPEAKER_FOLDER = f'/content/Mangio-RVC-Fork-Simple-CLI/logs/{speaker_name}'
WEIGHTS_FOLDER = '/content/Mangio-RVC-Fork-Simple-CLI/weights'
GOOGLE_DRIVE_PATH = '/content/drive/MyDrive/RVC_Backup'
GOOGLE_DRIVE_SPEAKER_PATH = f'{GOOGLE_DRIVE_PATH}/{speaker_name}'


def import_google_drive_backup():
    print(f"Importing Google Drive backup for speaker: {speaker_name}...")
    for root, dirs, files in os.walk(GOOGLE_DRIVE_SPEAKER_PATH):
        for filename in files:
            filepath = os.path.join(root, filename)
            backup_filepath = os.path.join(SPEAKER_FOLDER, os.path.relpath(filepath, GOOGLE_DRIVE_SPEAKER_PATH))
            backup_folderpath = os.path.dirname(backup_filepath)
            if not os.path.exists(backup_folderpath):
                os.makedirs(backup_folderpath)
                print(f'Created backup folder: {backup_folderpath}', flush=True)
            shutil.copy2(filepath, backup_filepath)  # copy file with metadata
            print(f'Imported file from Google Drive backup: {filename}')

    # Importing weights as it is, without speaker-specific logic
    weights_exist = False
    for root, dirs, files in os.walk(os.path.join(GOOGLE_DRIVE_PATH, 'weights')):
        for filename in files:
            if filename.endswith('.pth'):
                filepath = os.path.join(root, filename)
                weights_filepath = os.path.join(WEIGHTS_FOLDER, os.path.relpath(filepath, os.path.join(GOOGLE_DRIVE_PATH, 'weights')))
                weights_folderpath = os.path.dirname(weights_filepath)
                if not os.path.exists(weights_folderpath):
                    os.makedirs(weights_folderpath)
                    print(f'Created weights folder: {weights_folderpath}', flush=True)
                shutil.copy2(filepath, weights_filepath)  # copy file with metadata
                print(f'Imported file from weights: {filename}')
                weights_exist = True
    if weights_exist:
        print("Copied weights from Google Drive backup to local weights folder.")
    else:
        print("No weights found in Google Drive backup.")
    print("Google Drive backup import completed.")


def get_md5_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def copy_weights_folder_to_drive():
    destination_folder = os.path.join(GOOGLE_DRIVE_PATH, 'weights')
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    num_copied = 0
    for filename in os.listdir(WEIGHTS_FOLDER):
        if filename.endswith('.pth'):
            source_file = os.path.join(WEIGHTS_FOLDER, filename)
            destination_file = os.path.join(destination_folder, filename)
            if not os.path.exists(destination_file):
                shutil.copy2(source_file, destination_file)
                num_copied += 1
                print(f"Copied {filename} to Google Drive!")

    if num_copied == 0:
        print("No new finished models found for copying.")
    else:
        print(f"Finished copying {num_copied} files to Google Drive!")


def backup_files():
    print(f"\n Starting backup loop for speaker: {speaker_name}...")
    last_backup_timestamps_path = os.path.join(SPEAKER_FOLDER, 'last_backup_timestamps.txt')
    fully_updated = False  # boolean to track if all files are up to date
    try:
        with open(last_backup_timestamps_path, 'r') as f:
            last_backup_timestamps = dict(line.strip().split(':') for line in f)
    except:
        last_backup_timestamps = {}
    while True:
        updated = False  # flag to check if any files were updated
        for root, dirs, files in os.walk(SPEAKER_FOLDER):
            for filename in files:
                if filename != 'last_backup_timestamps.txt':
                    filepath = os.path.join(root, filename)
                    if os.path.isfile(filepath):
                        backup_filepath = os.path.join(GOOGLE_DRIVE_SPEAKER_PATH, os.path.relpath(filepath, SPEAKER_FOLDER))
                        backup_folderpath = os.path.dirname(backup_filepath)
                        if not os.path.exists(backup_folderpath):
                            os.makedirs(backup_folderpath)
                            print(f'Created backup folder: {backup_folderpath}', flush=True)
                        # check if file has changed since last backup
                        last_backup_timestamp = last_backup_timestamps.get(filepath)
                        current_timestamp = os.path.getmtime(filepath)
                        if last_backup_timestamp is None or float(last_backup_timestamp) < current_timestamp:
                            shutil.copy2(filepath, backup_filepath)  # copy file with metadata
                            last_backup_timestamps[filepath] = str(current_timestamp)  # update last backup timestamp
                            if last_backup_timestamp is None:
                                print(f'Backed up file: {filename}')
                            else:
                                print(f'Updating backed up file: {filename}')
                            updated = True
                            fully_updated = False  # if a file is updated, all files are not up to date
        # check if any files were deleted in Colab and delete them from the backup drive
        for filepath in list(last_backup_timestamps.keys()):
            if not os.path.exists(filepath):
                backup_filepath = os.path.join(GOOGLE_DRIVE_SPEAKER_PATH, os.path.relpath(filepath, SPEAKER_FOLDER))
                if os.path.exists(backup_filepath):
                    os.remove(backup_filepath)
                    print(f'Deleted file: {filepath}')
                del last_backup_timestamps[filepath]
                updated = True
                fully_updated = False  # if a file is deleted, all files are not up to date
        if not updated and not fully_updated:
            print("Files are up to date.")
            fully_updated = True  # if all files are up to date, set the boolean to True
            copy_weights_folder_to_drive()
            sleep_time = 15
        else:
            sleep_time = 0.1
        with open(last_backup_timestamps_path, 'w') as f:
            for filepath, timestamp in last_backup_timestamps.items():
                f.write(f'{filepath}:{timestamp}\n')
        time.sleep(sleep_time)  # wait for 15 seconds before checking again, or 1s if not fully up to date to speed up backups
