import datetime as dt
import json
import logging
import shutil
import sqlite3
import threading
import itertools
import torch
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from ultralytics.engine.results import Results
from core.pose_tracker import PoseTracker
from core.video_processor import VideoProcessor, VideoRecorder, ProcessingStats
from core.data_manager import DataManager
from utils.paths import PROJECT_ROOT, project_path

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
STAIRS = [{'id': 1, 'coordinates': [[10, 10], [200, 10], [200, 230], [10, 230]],
           'handrails': [{'id': 1, 'coordinates': [[10, 10], [30, 10], [30, 230], [10, 230]]}]},
          {'id': 2, 'coordinates': [[220, 10], [315, 10], [315, 230], [220, 230]],
           'handrails': [{'id': 1, 'coordinates': [[220, 10], [240, 10], [240, 230], [220, 230]]}]}]

def result(left=(70, 70, .9), right=(110, 70, .9), track_id=1, box=(40, 20, 150, 220)):
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, 9] = left
    keypoints[0, 10] = right
    boxes = np.array([[*box, track_id, .95, 0]], dtype=np.float32)
    return Results(FRAME, path='', names={0: 'person'}, boxes=torch.from_numpy(boxes), keypoints=torch.from_numpy(keypoints))

@pytest.fixture
def tracker(monkeypatch):
    monkeypatch.setattr(PoseTracker, '_load_model', lambda self: setattr(
        self, 'model', SimpleNamespace(track=lambda *a, **kw: [result()], predictor=None)))
    p = PoseTracker('unused', device='cpu')
    p.set_zones(STAIRS)
    return p

def drive(p, values, sample=None):
    if sample is not None:
        p.model.track = lambda *a, **kw: [sample]
    return [p.process(FRAME, 1.5, t)[0] for t in values]

@pytest.mark.parametrize('clock_step', [.0001, 1, 100])
def test_timer_uses_video_time(tracker, monkeypatch, clock_step):
    # CPU/GPU speed and wall-clock jumps have no bearing on event duration.
    import core.pose_tracker as module
    wall = itertools.count(1000, clock_step)
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda: next(wall)))
    observed = drive(tracker, [i / 10 for i in range(41)])
    assert not observed[15]
    assert observed[16]
    assert observed[-1][0]['violation_time'] == pytest.approx(4.0)

@pytest.mark.parametrize('left,right,state', [
    ((20,70,.9), (110,70,.9), 'holding'),
    ((70,70,.9), (20,70,.9), 'holding'),
    ((20,70,.9), (0,0,0), 'holding'),
    ((70,70,.9), (0,0,0), 'unknown'),
    ((0,0,0), (0,0,0), 'unknown'),
    ((20,70,.4), (0,0,0), 'unknown'),
])
def test_hand_visibility(tracker, left, right, state):
    assert not any(drive(tracker, [0,1,2,3], result(left,right)))
    assert tracker.tracked_persons[1]['contact_state'] == state

def test_other_stair_rail_does_not_count(tracker):
    observed = drive(tracker, [0,1,2], result((230,70,.9),(100,70,.9)))
    assert observed[-1]
    assert observed[-1][0]['stair_zone_id'] == 1

def test_unknown_breaks_episode(tracker):
    drive(tracker,[0,.5,1])
    drive(tracker,[1.1],result((0,0,0),(0,0,0)))
    observed=drive(tracker,[1.2,2.0],result())
    assert not any(observed)
    assert tracker.tracked_persons[1]['violation_timer'] == pytest.approx(.8)

def test_lost_track_breaks_episode_and_current_presence(tracker):
    drive(tracker,[0,.5,1])
    tracker.model.track=lambda *a,**kw: []
    assert not tracker.process(FRAME,1.5,1.2)[0]
    assert tracker.current_track_ids == set()
    observed=drive(tracker,[1.5,2.0],result())
    assert not any(observed)
    assert tracker.tracked_persons[1]['violation_timer'] == pytest.approx(.5)

def test_stair_change_resets_timer(tracker):
    drive(tracker,[0,.5,1,1.6])
    observed=drive(tracker,[1.7,2.0],result(box=(250,20,310,220)))
    assert not any(observed)
    assert tracker.tracked_persons[1]['stair_zone_id'] == 2

def test_model_failure_is_not_reported_as_no_violations(tracker):
    def fail(*a,**kw):raise RuntimeError('model failed')
    tracker.model.track=fail
    with pytest.raises(RuntimeError,match='model failed'):
        tracker.process(FRAME,1.5,0)

def test_reset_clears_state_and_bytetrack(tracker):
    drive(tracker,[0,1,2])
    resets=[]
    tracker.model.predictor=SimpleNamespace(trackers=[SimpleNamespace(reset=lambda:resets.append(1))])
    tracker.reset_tracking()
    assert not tracker.tracked_persons and not tracker.current_track_ids
    assert resets == [1]

@pytest.fixture
def manager(tmp_path):
    return DataManager(str(tmp_path/'analytics.db'))

def test_migrate_existing_database_preserves_data_and_backup(tmp_path):
    path=tmp_path/'legacy.db'
    with sqlite3.connect(path) as c:
        c.executescript('''CREATE TABLE cameras(id TEXT PRIMARY KEY, location TEXT, logo_path TEXT);
        INSERT INTO cameras VALUES ('old','location','');
        CREATE TABLE processed_videos(id INTEGER PRIMARY KEY AUTOINCREMENT,camera_id TEXT,filename TEXT,
        start_time TIMESTAMP,end_time TIMESTAMP,duration REAL,processing_time TIMESTAMP,
        total_persons INTEGER,violation_persons INTEGER);
        INSERT INTO processed_videos VALUES (1,'old','old.mp4','2025-01-01','2025-01-02',86400,'2025-01-02',2,1);
        CREATE TABLE violations(id INTEGER PRIMARY KEY AUTOINCREMENT,video_id INTEGER,track_id INTEGER,
        timestamp TIMESTAMP,duration REAL,violation_file TEXT);
        INSERT INTO violations VALUES (1,1,7,'2025-01-01',3,'clip.avi');''')
    m=DataManager(str(path))
    assert m.get_cameras()[0]['id']=='old'
    assert m.get_videos_by_camera('old')[0]['total_persons']==2
    with sqlite3.connect(path) as c:
        assert c.execute('select global_id from violations').fetchone()[0]==7
        assert c.execute('pragma user_version').fetchone()[0]==2
    backups=list(tmp_path.glob('*.bak'))
    assert len(backups)==1
    with sqlite3.connect(backups[0]) as c:
        assert 'global_id' not in {r[1] for r in c.execute('pragma table_info(violations)')}
    DataManager(str(path))
    assert len(list(tmp_path.glob('*.bak')))==1


def saved_input():
    start=dt.datetime(2025,1,1)
    personnel={1:dict(first_seen=start,last_seen=start+dt.timedelta(seconds=5),frames_count=51,
                      first_frame_index=1,last_frame_index=51,had_violation=True,bbox=[1,2,3,4])}
    episodes=[dict(track_id=8,global_id=1,timestamp=start,duration=4,stair_zone_id=1,violation_file='clip.avi')]
    return start,personnel,episodes

def save(m,personnel=None,episodes=None):
    start,p,e=saved_input()
    return m.save_analysis({'camera_id':'test','location':'test'},'file.avi',start,start+dt.timedelta(seconds=6),
        ProcessingStats(frames_processed=60,avg_fps=30), p if personnel is None else personnel,
        e if episodes is None else episodes,123,'320x240')

def test_repeat_save_replaces_results_without_double_count(manager):
    first=save(manager)
    assert save(manager)==first
    with manager.get_connection() as c:
        assert c.execute('select count(*) from processed_videos').fetchone()[0]==1
        assert c.execute('select count(*) from violations').fetchone()[0]==1
        assert tuple(c.execute('select total_persons,total_violations,videos_processed from daily_stats').fetchone())==(1,1,1)
    save(manager,{},[])
    with manager.get_connection() as c:
        assert c.execute('select count(*) from personnel_records').fetchone()[0]==0
        assert c.execute('select count(*) from violations').fetchone()[0]==0
        assert tuple(c.execute('select total_persons,total_violations,videos_processed from daily_stats').fetchone())==(0,0,1)

def test_failed_save_rolls_back_previous_results(manager):
    save(manager)
    _,p,e=saved_input();del e[0]['duration']
    with pytest.raises(KeyError):save(manager,p,e)
    with manager.get_connection() as c:
        assert c.execute('select duration from violations').fetchone()[0]==4
        assert c.execute('select count(*) from personnel_records').fetchone()[0]==1

def test_recorder_cleanup_finishes_and_clip_is_readable(tmp_path):
    r=VideoRecorder(str(tmp_path))
    r.start_recording(1,dt.datetime(2025,1,1),320,240,10,'test')
    for _ in range(5):r.write_frame(1,FRAME,True)
    done=threading.Thread(target=r.cleanup,daemon=True);done.start();done.join(2)
    assert not done.is_alive()
    clip=next(tmp_path.glob('*.avi'))
    cap=cv2.VideoCapture(str(clip))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT)==5
    cap.release()
    r.cleanup()

def make_video(path,frames=60):
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240))
    assert w.isOpened()
    for _ in range(frames):w.write(FRAME)
    w.release()

@pytest.fixture
def processor(tmp_path,monkeypatch,tracker,manager):
    import core.video_processor as module
    monkeypatch.setattr(module,'DataManager',lambda:manager)
    monkeypatch.setattr(module,'PoseTracker',lambda *a,**kw:tracker)
    dummy_model=tmp_path/'model.pt';dummy_model.write_bytes(b'test placeholder')
    cfg={'camera_id':'test','location':'test','yolo_model_path':str(dummy_model),
         'violations_folder_path':str(tmp_path/'clips'),'stair_zones':STAIRS,'violation_timer_threshold':1.5}
    p=VideoProcessor(cfg)
    p.motion_detector.detect=lambda frame:(True,frame)
    p.enable_visualization=False
    return p

def run_video(p,path):
    events=[];p.process_video_file(str(path),events.append);p.processing_thread.join(10)
    assert not p.processing_thread.is_alive()
    return events

def test_video_pipeline_records_one_episode_and_repeat_is_stable(processor,tmp_path,manager):
    video=tmp_path/'test 2025-01-01 00-00-00_000.avi';make_video(video)
    for _ in range(2):
        events=run_video(processor,video)
        assert events[-1]['status']=='completed',events[-1]
        assert events[-1]['total_frames']==60
        assert events[-1]['violation_persons']==1
        with manager.get_connection() as c:
            row=c.execute('select * from violations').fetchall()
            assert len(row)==1 and row[0]['duration']==pytest.approx(4.9)
            clip=Path(processor.camera_config['violations_folder_path'])/row[0]['violation_file']
            assert clip.exists()
            cap=cv2.VideoCapture(str(clip));assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==50;cap.release()
            assert c.execute('select videos_processed from daily_stats').fetchone()[0]==1
        assert not processor.video_recorder.frame_buffers
    processor.cleanup()

def test_cancellation_leaves_worker_resources_owned_and_no_complete_record(processor,tmp_path,manager):
    video=tmp_path/'test.avi';make_video(video)
    entered=threading.Event();release=threading.Event()
    original=processor._process_single_frame
    def blocked(*args):
        entered.set();assert release.wait(3);return original(*args)
    processor._process_single_frame=blocked
    events=[];processor.process_video_file(str(video),events.append)
    assert entered.wait(2)
    assert processor.stop_processing(wait=False) is False
    assert processor.video_cap.isOpened()
    release.set();processor.processing_thread.join(5)
    assert not processor.processing_thread.is_alive()
    assert events[-1]['status']=='stopped'
    assert processor.video_cap is None
    with manager.get_connection() as c:
        assert c.execute('select count(*) from processed_videos').fetchone()[0]==0

def test_invalid_video_reports_error(processor,tmp_path):
    events=run_video(processor,tmp_path/'missing.avi')
    assert events[-1]['status']=='error'
    assert not processor.is_processing

def test_bundle_paths():
    assert project_path('C:/OKO_PORUCHNI/models/model.pt')==str(PROJECT_ROOT/'models/model.pt')
    assert project_path(r'C:\OKO_PORUCHNI\models\model.pt')==str(PROJECT_ROOT/'models/model.pt')
    assert project_path('D:/archive/video.mp4')=='D:/archive/video.mp4'
    assert project_path('/private/tmp/video.mp4')=='/private/tmp/video.mp4'

def test_two_people_are_independent(tracker):
    points = np.zeros((2,17,3),dtype=np.float32)
    points[0,9]=[20,70,.9];points[0,10]=[80,70,.9]
    points[1,9]=[80,90,.9];points[1,10]=[110,90,.9]
    boxes=torch.tensor([[40,20,150,220,1,.95,0],[50,20,160,220,2,.95,0]])
    sample=Results(FRAME,path='',names={0:'person'},boxes=boxes,keypoints=torch.from_numpy(points))
    observed=drive(tracker,[0,1,2],sample)
    assert [v['track_id'] for v in observed[-1]] == [2]

def test_missing_handrail_configuration_is_unknown(tracker):
    tracker.set_zones([dict(STAIRS[0],handrails=[])])
    assert not any(drive(tracker,[0,1,2,3]))
    assert tracker.tracked_persons[1]['contact_state']=='unknown'

def test_configuration_paths_remain_portable_after_edit(tmp_path,monkeypatch):
    from config.camera_config import CameraConfig
    import utils.paths as paths
    bundle=tmp_path/'bundle'
    (bundle/'data/videos/ВК3.4').mkdir(parents=True)
    monkeypatch.setattr(paths,'PROJECT_ROOT',bundle)
    path=tmp_path/'cameras.json'
    path.write_text(json.dumps({'cameras':[
        {'camera_id':'A','video_folder_path':'C:/OKO_PORUCHNI/data/videos/BK3.4','yolo_model_path':'models/model.pt'},
        {'camera_id':'B','video_folder_path':'C:/OKO_PORUCHNI/data/videos/BK1.9','yolo_model_path':'models/model.pt'}]}))
    manager=CameraConfig(str(path))
    a=manager.get_camera('A')
    assert Path(a['video_folder_path']).name=='ВК3.4'
    a['description']='edited';manager.add_camera(a)
    saved=json.loads(path.read_text())['cameras']
    assert all(not row['video_folder_path'].startswith(str(bundle)) for row in saved)
    assert manager.get_camera('B')['yolo_model_path']==str(bundle/'models/model.pt')

def test_recording_keeps_frames_during_short_detection_gap(processor,tmp_path,manager):
    video=tmp_path/'test 2025-01-01 00-00-00_000.avi';make_video(video)
    calls=[0]
    def inference(*a,**kw):
        calls[0]+=1
        return [] if 30 <= calls[0] <= 34 else [result()]
    processor.pose_tracker.model.track=inference
    events=run_video(processor,video)
    assert events[-1]['status']=='completed'
    with manager.get_connection() as c:
        clips={r[0] for r in c.execute('select violation_file from violations')}
    assert len(clips)==1
    cap=cv2.VideoCapture(str(Path(processor.camera_config['violations_folder_path'])/next(iter(clips))))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==50
    cap.release()
