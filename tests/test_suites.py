from independent_mri.suite import recipes


def test_split_ablation_is_complete_and_matched():
    all_jobs = recipes('ablation')
    assert recipes('ablation-a')+recipes('ablation-b')==all_jobs
    assert len(all_jobs)==12
    assert len(set(job[0] for job in all_jobs))==12
    assert {job[5] for job in all_jobs}=={6}
    assert {job[6] for job in all_jobs}=={-1}
    policies = {job[2] for job in all_jobs}
    assert policies=={'L1','L2','L3','L4'}


def test_warm_and_clean_teachers_are_separate():
    assert {job[5] for job in recipes('warm')}=={12}
    assert {job[6] for job in recipes('teachers')}=={0,1,2}
