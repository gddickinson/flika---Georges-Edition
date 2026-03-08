import os
import pytest

# Marker: tests in these modules don't need a full FlikaApplication.
_STANDALONE_MODULES = {
    'test_io_registry', 'test_settings_batch',
    'test_macro_recorder', 'test_undo',
    'test_spt_detection', 'test_spt_linking', 'test_spt_features',
    'test_spt_io', 'test_spt_pipeline',
    'test_ai',
}

_flikaApp = None

def _get_app():
    global _flikaApp
    if _flikaApp is None:
        from ..app.application import FlikaApplication
        _flikaApp = FlikaApplication()
        # Replace g.alert with a non-blocking version during testing
        from .. import global_vars as g
        from ..logger import logger
        def _test_alert(msg, title="flika - Alert"):
            logger.info('Test alert (auto-dismissed): %s', msg)
            if g.m is not None:
                g.m.statusBar().showMessage(msg)
        g.alert = _test_alert
    return _flikaApp

@pytest.fixture(scope='module', autouse=True)
def fa(request):
    # Skip app creation for standalone test modules that don't need Qt/plugins
    module_name = os.path.splitext(os.path.basename(request.fspath))[0]
    if module_name in _STANDALONE_MODULES:
        yield None
        return
    app = _get_app()
    yield app