from google.cloud import storage


class GoogleCloudStorage:
    """
    See https://github.com/googleapis/python-storage/tree/main/samples
    for snippets
    """

    @staticmethod
    def _upload_blob(
        bucket_name: str, source_file_name: str, destination_blob_name: str
    ) -> None:
        """Uploads a file to the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

    def upload(self, source_file_name: str, destination_name: str) -> None:
        pass
