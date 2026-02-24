import dropbox
import os
import sys
from tqdm import tqdm
from dropbox.exceptions import ApiError

# Replace with your Dropbox API access token
ACCESS_TOKEN = "sl.u.AGWCRBXlOvkjwj4P8HnJgwzpNUERtRqLJq9d0_xlVe54cZKzspsHtHkVW8wBoc6OWNeYAPtx9cEVIF_W_zBl6AAlEgUN5GJ7jroyZ9PteJoqP-q_jAbAxVLuDZH1XdyL-PC9sggGgu_NT2HnvJVQcRiiTd3Z55pqKnTImdAvvWf3JZxsQXwmIptlJDT7FbmkzqGlfXkQqQcCk_aGkiSo5gA7l-R6SHCM-Kjcv6nAF3QfQYEhWxQQbZ1DiQPG036kwngOqB-nflO0gjYKXNAHFjOvjWjgBTjin4R7lKAqNKDzW7oK50y7LtOJE_MoUHSDwa_l5Up7XFogezqSHYQ0Dk2izp1a7bD_PqSls2dpvE9A8pBObC7KpOpYfolJoi78z7BswCp9Q6jtdT0_pMOv5U0YheRphpFmApQ0t-p64bYUO5bLlSWYVEzMY-nroCMt4i2Aul5ThG1C0McpT5BoBXzMH20G7b-OwPtjmxIBmjt93Rs3vWEhBgfBDUuFYsQ0qQCFCz0jHt-98Q-gmz6kjeIoX2UFNzwCl7_9SEUrwqP8Y5UYHUYZearpTkdX0SZXQpmzaSZ3dUlZMPGtDcVVXwGcNs_aVE-G-q68Hsg3rSbC0oXFpoJSz5T5_FiZMJu7nFNYZLaMvNE5iCpd8rJmGL72h8qUVgu3ADSIJZqPWoPyozjLVE5Z7YSRmHdzShUcw3HiZOG5GlnufSCeT0tFlEAtefcGl6zNvxPvp6RiIOSYDqOCltzUYzqLhLXTw2l0fA0JJMdGcD9qnrBMrPd0GWStnvS9yfiqbYnNnm9LQYDOIaFURnl5-AYtLfNlDM8p9eUgm9Cs56baAhmbsIeBJlnS_5SSd_0D-XBDfYahhI1urbnczyZ-2t37pjoveH_SJ-_CtJfs0DwXjAqmtDV_DVB8iIlZYMWiT10Tkh1I_iaMBjIegUzXDfg0vhKWRrgNy6N83wvGkI7CTrt2vSe5cVHm1Tp9il_dTlk7g2IJdtMBivugyejUdLVQ6hxSEE8UnNPrGZmxF6AW73g_Z2zhBw3WkaKM3t6yzFspVl-Wtu_S_tSc5v1aUDm8-oZLOHY6Lk1rcTnHa3slcOjWMJpJk1DDiB3nwcge-wC4VfKy_86meU96DObCIZLI1hdzdnp9Q-e9qqNHS99WGmrei7XnCtLym9MTMiSs99K6KG06WyFhJD-YAWdsqZ617yLiOURJ2ONb797uGrBT1pxs7PzmwbgKhqcdoqnITEtjk4nwCQ7sDtgNjf8KNdsQ5QxKtTTYnsb52LzKlNSLBX5OK0qj-fE2iXpPty7WucJZ876hjFKwiQ"

# Path of the folder in Dropbox (e.g., "/SharedFolder")
DROPBOX_FOLDER_PATH = "/report2"

# Local folder to save files
LOCAL_DEST = "simulation/"

os.makedirs(LOCAL_DEST, exist_ok=True)

# Connect to Dropbox
dbx = dropbox.Dropbox(ACCESS_TOKEN)

def download_folder(dropbox_path, local_path):
    try:
        result = dbx.files_list_folder(dropbox_path)
    except ApiError as e:
        print(f"Error listing folder {dropbox_path}: {e}")
        return

    entries = result.entries
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)

    for entry in tqdm(entries):
        entry_path = entry.path_lower
        local_file_path = os.path.join(local_path, entry.name)

        if isinstance(entry, dropbox.files.FileMetadata):
            os.makedirs(local_path, exist_ok=True)
            # print(f"Downloading file: {entry_path} → {local_file_path}")
            try:
                dbx.files_download_to_file(local_file_path, entry_path)
            except Exception as e:
                print(f"Failed to download {entry_path}: {e}")
        elif isinstance(entry, dropbox.files.FolderMetadata):
            new_local_folder = os.path.join(local_path, entry.name)
            download_folder(entry_path, new_local_folder)

if __name__ == "__main__":
    print(f"Starting download of {DROPBOX_FOLDER_PATH} to {LOCAL_DEST}")
    download_folder(DROPBOX_FOLDER_PATH, LOCAL_DEST)
    print("Download complete!")