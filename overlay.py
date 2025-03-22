import win32gui
import win32con
import win32api

class Overlay:
    def __init__(self, width=1920, height=1080, alpha=200):
        self.hInstance = win32api.GetModuleHandle()
        self.className = "OverlayWindowClass"
        self.hWnd = None
        self.width = width
        self.height = height
        self.alpha = alpha
        self.visible = False

        self._register_class()
        self._create_window()

    def _register_class(self):
        wndClass = win32gui.WNDCLASS()
        wndClass.lpfnWndProc = self._wnd_proc
        wndClass.hInstance = self.hInstance
        wndClass.lpszClassName = self.className
        wndClass.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)
        wndClass.hbrBackground = win32con.COLOR_WINDOW
        win32gui.RegisterClass(wndClass)

    def _create_window(self):
        screen_w = win32api.GetSystemMetrics(0)
        screen_h = win32api.GetSystemMetrics(1)

        self.hWnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_TRANSPARENT,
            self.className,
            None,
            win32con.WS_POPUP,
            0,
            0,
            screen_w,
            screen_h,
            None,
            None,
            self.hInstance,
            None
        )

        win32gui.SetLayeredWindowAttributes(self.hWnd, 0x000000, self.alpha, win32con.LWA_ALPHA)
        win32gui.ShowWindow(self.hWnd, win32con.SW_SHOW)

    def toggle(self):
        self.visible = not self.visible
        win32gui.ShowWindow(self.hWnd, win32con.SW_SHOW if self.visible else win32con.SW_HIDE)

    def update(self, bounding_boxes):
        hdc = win32gui.GetDC(self.hWnd)
        pen = win32gui.CreatePen(win32con.PS_SOLID, 2, win32api.RGB(0, 255, 0))
        win32gui.SelectObject(hdc, pen)

        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            win32gui.Rectangle(hdc, x1, y1, x2, y2)

        win32gui.DeleteObject(pen)
        win32gui.ReleaseDC(self.hWnd, hdc)

    def _wnd_proc(self, hWnd, msg, wParam, lParam):
        if msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
        return win32gui.DefWindowProc(hWnd, msg, wParam, lParam)
