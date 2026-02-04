#!/usr/bin/env python3
"""
Interactive Zone Configuration Tool
Helps you define accurate zones for your CCTV footage
"""

import cv2
import numpy as np
import argparse
import json


class ZoneConfigurator:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.zones = {}
        self.current_zone = None
        self.points = []
        self.frame = None
        self.original_frame = None

        # Load first frame
        cap = cv2.VideoCapture(video_path)
        ret, self.original_frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        self.frame = self.original_frame.copy()
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.redraw()

    def redraw(self):
        self.frame = self.original_frame.copy()

        # Draw existing zones
        for zone_name, zone_data in self.zones.items():
            polygon = np.array(zone_data['polygon'], dtype=np.int32)
            color = tuple(zone_data['color'])

            # Border
            cv2.polylines(self.frame, [polygon], True, color, 2)

            # Semi-transparent overlay
            overlay = self.frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            self.frame = cv2.addWeighted(overlay, 0.30, self.frame, 0.70, 0)

            # Label
            centroid = polygon.mean(axis=0).astype(int)
            cv2.putText(
                self.frame,
                zone_name,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Draw current points
        for i, point in enumerate(self.points):
            cv2.circle(self.frame, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(
                self.frame,
                f"P{i+1}",
                (point[0] + 10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Draw current polygon preview
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(
                self.frame,
                [pts],
                len(self.points) >= 3,
                (0, 255, 255),
                2
            )

        # Instructions
        self.draw_instructions()

        cv2.imshow("Zone Configurator", self.frame)

    def draw_instructions(self):
        instructions = [
            f"Current Zone: {self.current_zone or 'None'}",
            f"Points: {len(self.points)}/4+",
            "",
            "Left Click: Add point",
            "Right Click: Remove last point",
            "Press 's': Save current zone",
            "Press 'n': New zone",
            "Press 'c': Clear points",
            "Press 'f': Finish & export",
            "Press 'q': Quit"
        ]

        y_offset = 30
        for i, text in enumerate(instructions):
            color = (255, 255, 255) if i < 2 else (200, 200, 200)
            cv2.putText(
                self.frame,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    def save_zone(self):
        if not self.current_zone:
            print("⚠️  No zone name set. Press 'n' to create new zone.")
            return

        if len(self.points) < 3:
            print("⚠️  Need at least 3 points for a zone.")
            return

        # Choose color for zone
        colors = {
            'counter': (0, 255, 255),
            'behind_counter': (0, 165, 255),
            'customer_area': (255, 0, 0),
            'entrance': (0, 255, 0),
            'seating': (255, 0, 255),
            'kitchen': (128, 0, 128),
            'storage': (0, 128, 128)
        }

        color = colors.get(self.current_zone, (128, 128, 128))

        self.zones[self.current_zone] = {
            'polygon': self.points.copy(),
            'color': color
        }

        print(f"✅ Saved zone: {self.current_zone} with {len(self.points)} points")
        self.points = []
        self.current_zone = None
        self.redraw()

    def new_zone(self):
        zone_name = input(
            "\n📝 Enter zone name (e.g., 'counter', 'behind_counter', 'entrance'): "
        ).strip()

        if zone_name:
            self.current_zone = zone_name
            self.points = []
            print(f"✏️  Creating zone: {zone_name}")
            print("   Click to add polygon points on the video frame.")

        self.redraw()

    def export_config(self, output_path: str = "zone_config.py"):
        if not self.zones:
            print("⚠️  No zones to export!")
            return

        # Generate Python config safely (NO .format() issues)
        config_text = '''"""
Auto-generated Zone Configuration
"""
import numpy as np

ZONES = {
'''

        for zone_name, zone_data in self.zones.items():
            polygon = zone_data['polygon']
            color = zone_data['color']

            # Determine weights based on zone name
            if 'behind' in zone_name or 'staff' in zone_name or 'kitchen' in zone_name:
                emp_weight, cust_weight = 1.5, 0.0
            elif 'counter' in zone_name:
                emp_weight, cust_weight = 1.0, 0.3
            elif 'customer' in zone_name or 'seating' in zone_name:
                emp_weight, cust_weight = 0.2, 1.0
            else:
                emp_weight, cust_weight = 0.5, 0.5

            config_text += f'''    "{zone_name}": {{
        "polygon": np.array([
'''

            for point in polygon:
                config_text += f'            {point},\n'

            config_text += f'''        ]),
        "color": {tuple(color)},
        "employee_weight": {emp_weight},
        "customer_weight": {cust_weight}
    }},
'''

        # IMPORTANT FIX: use f-string + escape literal closing brace
        config_text += f'''}}

# Video resolution
VIDEO_WIDTH = {self.width}
VIDEO_HEIGHT = {self.height}
'''

        # Save to file
        with open(output_path, 'w', encoding="utf-8") as f:
            f.write(config_text)

        print(f"\n✅ Configuration exported to: {output_path}")
        print("\n📋 Copy this to your config_mvp.py file:")
        print("=" * 60)
        print(config_text)
        print("=" * 60)

        # Save JSON backup
        json_output = {
            'zones': self.zones,
            'video_width': self.width,
            'video_height': self.height
        }

        json_path = output_path.replace('.py', '.json')
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)

        print(f"✅ JSON backup saved to: {json_path}")

    def run(self, output_path: str = "zone_config.py"):
        print("\n" + "=" * 60)
        print("🎯 ZONE CONFIGURATION TOOL")
        print("=" * 60)
        print(f"Video: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print("\nRecommended zones for retail/coffee shop:")
        print("  • counter - Main service counter")
        print("  • behind_counter - Staff-only area behind counter")
        print("  • customer_area - Seating/waiting area")
        print("  • entrance - Entry/exit zone")
        print("=" * 60 + "\n")

        cv2.namedWindow("Zone Configurator")
        cv2.setMouseCallback("Zone Configurator", self.mouse_callback)

        self.redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('n'):
                self.new_zone()

            elif key == ord('s'):
                self.save_zone()

            elif key == ord('c'):
                self.points = []
                self.redraw()
                print("🔄 Points cleared")

            elif key == ord('f'):
                self.export_config(output_path)
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive zone configuration tool")
    parser.add_argument("--video", type=str, required=True, help="Path to CCTV video")
    parser.add_argument("--output", type=str, default="zone_config.py",
                        help="Output configuration file")
    args = parser.parse_args()

    configurator = ZoneConfigurator(args.video)
    configurator.run(args.output)


if __name__ == "__main__":
    main()
