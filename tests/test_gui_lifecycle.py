import threading
import time
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QCloseEvent
from gui.main_window import VideoProcessingThread, CameraProcessorWindow, MainWindow
from core.data_manager import DataManager

@pytest.fixture(scope='module')
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def wait_for(app, predicate, timeout=3):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents()
        if predicate():return
        time.sleep(.005)
    assert predicate()

class BlockingProcessor:
    def __init__(self):
        self.entered=threading.Event()
        self.release=threading.Event()
        self.processing_thread=None
        self.is_processing=False
        self.cleaned=False
    def process_video_file(self, path, callback):
        self.is_processing=True
        def work():
            callback({'status':'started','timestamp':'2025-01-01 00:00:00'})
            self.entered.set()
            self.release.wait(3)
            self.is_processing=False
            callback({'status':'stopped' if getattr(self,'stopped',False) else 'completed'})
        self.processing_thread=threading.Thread(target=work)
        self.processing_thread.start()
    def stop_processing(self, wait=False):
        self.stopped=True
        self.release.set()
    def cleanup(self):
        assert not self.is_processing
        self.cleaned=True
        return True

@pytest.mark.parametrize('stop',[False,True])
def test_qthread_waits_for_worker_and_preserves_video_timestamp(app,stop):
    p=BlockingProcessor();q=VideoProcessingThread(p,'unused');events=[];finished=[]
    q.progress_update.connect(events.append);q.finished.connect(lambda:finished.append(True))
    q.start()
    wait_for(app,p.entered.is_set)
    assert q.isRunning() and not finished
    if stop:q.stop()
    else:p.release.set()
    wait_for(app,lambda:not q.isRunning());app.processEvents()
    assert finished and q.outcome==('stopped' if stop else 'completed')
    assert events[0]['timestamp']=='2025-01-01 00:00:00'
    q.wait()

@pytest.fixture
def window(app, tmp_path, monkeypatch):
    import gui.main_window as module
    manager=DataManager(str(tmp_path/'analytics.db'))
    monkeypatch.setattr(module,'DataManager',lambda:manager)
    config={'camera_id':'test','location':'test','video_folder_path':str(tmp_path),
            'violations_folder_path':str(tmp_path/'clips'),'violation_timer_threshold':1.5}
    w=CameraProcessorWindow(camera_config=config)
    yield w
    if w.processing_thread and w.processing_thread.isRunning():
        w.processing_thread.stop();w.processing_thread.wait(3000)
    w.close();app.processEvents()

def test_camera_window_keeps_progress_until_worker_stops(app,window):
    p=BlockingProcessor();window.video_processor=p
    window._start_video_processing('unused','unused')
    wait_for(app,p.entered.is_set)
    assert not window.process_btn.isEnabled()
    assert window.stop_btn.isEnabled()
    assert not window.progress_bar.isHidden()
    window.stop_processing()
    wait_for(app,lambda:window.process_btn.isEnabled())
    assert window.status_label.text()=='Обработка остановлена'
    assert not window.stop_btn.isEnabled()

def test_camera_close_waits_for_worker(app,window,monkeypatch):
    p=BlockingProcessor();window.video_processor=p
    window._start_video_processing('unused','unused')
    wait_for(app,p.entered.is_set)
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**kw:QMessageBox.Yes)
    event=QCloseEvent();window.closeEvent(event)
    assert not event.isAccepted()
    wait_for(app,lambda:p.cleaned)
    assert not window.processing_thread.isRunning()

def test_error_label_is_not_replaced_with_success(app,window,monkeypatch):
    monkeypatch.setattr(QMessageBox,'critical',lambda *a,**kw:None)
    window.processing_thread=SimpleNamespace(outcome='error',isRunning=lambda:False)
    window.handle_processing_error('test error')
    window.processing_finished()
    assert 'ошибкой' in window.status_label.text()
    assert 'test error' in window.process_status_label.text()
