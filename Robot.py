class Robot:
    def __init__(self, qr_data, points, current_frame):
        self.qr_data = qr_data
        self.points = points
        self.last_seen = current_frame

    def update(self, new_points, frame_id):
        self.points = new_points
        self.last_seen = frame_id

    def is_active(self, current_frame, max_lost=30):
        return (current_frame - self.last_seen) <= max_lost