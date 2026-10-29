from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from fetch_artifact import replace_with_retry,retry_delay


class ArtifactFetch(unittest.TestCase):
    def test_retry_after_seconds(self):
        self.assertEqual(retry_delay('120'),120)
        self.assertEqual(retry_delay(None),60)

    def test_retry_after_date(self):
        now = datetime(2026,10,22,4,0,tzinfo=timezone.utc)
        self.assertEqual(retry_delay('Thu, 22 Oct 2026 04:02:00 GMT',now),120)

    def test_atomic_replace_retries_transient_windows_lock(self):
        with TemporaryDirectory() as directory:
            source=Path(directory)/'download.unique.part'
            destination=Path(directory)/'manifest.json'
            source.write_text('manifest')
            real_replace=Path.replace
            calls=0

            def flaky_replace(path,target):
                nonlocal calls
                calls+=1
                if calls==1:
                    raise PermissionError('simulated transient Windows sharing violation')
                return real_replace(path,target)

            from unittest.mock import patch
            with patch.object(Path,'replace',flaky_replace):
                replace_with_retry(source,destination,attempts=2,base_delay=0,sleep=lambda _:None)
            self.assertEqual(calls,2)
            self.assertEqual(destination.read_text(),'manifest')
            self.assertFalse(source.exists())

    def test_atomic_replace_surfaces_a_persistent_lock_as_retryable_error(self):
        with TemporaryDirectory() as directory:
            source=Path(directory)/'download.unique.part'
            destination=Path(directory)/'manifest.json'
            source.write_text('manifest')
            from unittest.mock import patch
            with patch.object(Path,'replace',side_effect=PermissionError('persistent lock')):
                with self.assertRaisesRegex(RuntimeError,'after 2 attempts'):
                    replace_with_retry(source,destination,attempts=2,base_delay=0,sleep=lambda _:None)


if __name__=='__main__': unittest.main()
