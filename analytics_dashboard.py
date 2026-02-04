#!/usr/bin/env python3
"""
Real-time Analytics Dashboard for Retail Store
Employee Performance, Customer Analytics, and Store Metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict
import cv2


class RetailAnalytics:
    """Comprehensive analytics for retail store performance"""

    def __init__(self, results_csv: str, video_fps: float = 30.0):
        self.df = pd.read_csv(results_csv)
        self.fps = video_fps
        self.video_duration = (
            self.df["total_frames"].max() / video_fps if len(self.df) > 0 else 0
        )

    def employee_performance_metrics(self) -> Dict:
        """Calculate employee performance metrics"""
        employees = self.df[self.df["class"] == "Employee"]

        if len(employees) == 0:
            return {
                "total_employees": 0,
                "avg_shift_duration": 0,
                "total_work_time": 0,
                "counter_utilization": 0,
                "behind_counter_time": 0,
                "mobility_score": 0,
                "employee_ids": [],
                "employees": [],
            }

        metrics = {
            "total_employees": len(employees),
            "avg_shift_duration": employees["duration_sec"].mean(),
            "total_work_time": employees["duration_sec"].sum(),
            "counter_utilization": employees["zone_counter_ratio"].mean() * 100,
            "behind_counter_time": employees["zone_behind_counter_ratio"].mean() * 100,
            "mobility_score": employees["avg_speed"].mean(),
            "employee_ids": employees["track_id"].tolist(),
            "employees": [],
        }

        for _, emp in employees.iterrows():
            metrics["employees"].append(
                {
                    "id": int(emp["track_id"]),
                    "duration_min": emp["duration_sec"] / 60,
                    "counter_time_pct": emp["zone_counter_ratio"] * 100,
                    "behind_counter_time_pct": emp["zone_behind_counter_ratio"] * 100,
                    "customer_area_time_pct": emp["zone_customer_area_ratio"] * 100,
                    "avg_speed": emp["avg_speed"],
                    "confidence": emp["confidence"],
                }
            )

        return metrics

    def customer_analytics(self) -> Dict:
        """Calculate customer analytics"""
        customers = self.df[self.df["class"] == "Customer"]

        if len(customers) == 0:
            return {
                "total_customers": 0,
                "avg_visit_duration": 0,
                "median_visit_duration": 0,
                "total_foot_traffic": 0,
                "conversion_rate": 0,
                "avg_browsing_time": 0,
                "entrance_usage": 0,
                "counter_interaction": 0,
                "avg_movement_speed": 0,
                "path_directness": 0,
                "quick_visits": 0,
                "short_visits": 0,
                "medium_visits": 0,
                "long_visits": 0,
                "browsers": 0,
                "quick_buyers": 0,
            }

        metrics = {
            "total_customers": len(customers),
            "avg_visit_duration": customers["duration_sec"].mean(),
            "median_visit_duration": customers["duration_sec"].median(),
            "total_foot_traffic": len(customers),
            "avg_browsing_time": (
                customers["zone_customer_area_ratio"] * customers["duration_sec"]
            ).mean(),
            "entrance_usage": customers["zone_entrance_ratio"].mean() * 100,
            "counter_interaction": customers["zone_counter_ratio"].mean() * 100,
            "avg_movement_speed": customers["avg_speed"].mean(),
            "path_directness": customers["straightness"].mean(),
            "quick_visits": len(customers[customers["duration_sec"] < 10]),
            "short_visits": len(
                customers[
                    (customers["duration_sec"] >= 10) & (customers["duration_sec"] < 30)
                ]
            ),
            "medium_visits": len(
                customers[
                    (customers["duration_sec"] >= 30) & (customers["duration_sec"] < 60)
                ]
            ),
            "long_visits": len(customers[customers["duration_sec"] >= 60]),
            "browsers": len(customers[customers["zone_customer_area_ratio"] > 0.6]),
            "quick_buyers": len(customers[customers["straightness"] > 0.7]),
        }

        return metrics

    def zone_utilization(self) -> Dict:
        """Calculate zone utilization metrics"""
        total_person_time = self.df["duration_sec"].sum()

        zones = ["behind_counter", "counter", "customer_area", "entrance"]
        utilization = {}

        for zone in zones:
            zone_col = f"zone_{zone}_ratio"
            if zone_col in self.df.columns:
                zone_time = (self.df[zone_col] * self.df["duration_sec"]).sum()

                emp_df = self.df[self.df["class"] == "Employee"]
                cust_df = self.df[self.df["class"] == "Customer"]

                emp_time = (
                    (emp_df[zone_col] * emp_df["duration_sec"]).sum()
                    if len(emp_df) > 0
                    else 0
                )
                cust_time = (
                    (cust_df[zone_col] * cust_df["duration_sec"]).sum()
                    if len(cust_df) > 0
                    else 0
                )

                utilization[zone] = {
                    "total_time_sec": float(zone_time),
                    "percent_of_total": (zone_time / total_person_time * 100)
                    if total_person_time > 0
                    else 0,
                    "employee_time_sec": float(emp_time),
                    "customer_time_sec": float(cust_time),
                    "employee_pct": (emp_time / zone_time * 100) if zone_time > 0 else 0,
                    "customer_pct": (cust_time / zone_time * 100) if zone_time > 0 else 0,
                }

        return utilization

    def traffic_patterns(self) -> Dict:
        """Analyze traffic patterns over time (simple bins)"""
        bin_size = 300  # 5 minutes
        num_bins = int(self.video_duration / bin_size) + 1

        patterns = {
            "time_bins": [],
            "employee_count": [],
            "customer_count": [],
            "total_count": [],
        }

        for i in range(num_bins):
            start_time = i * bin_size
            end_time = (i + 1) * bin_size

            employees_in_bin = len(self.df[(self.df["class"] == "Employee")])
            customers_in_bin = len(self.df[(self.df["class"] == "Customer")])

            patterns["time_bins"].append(f"{int(start_time/60)}-{int(end_time/60)}min")
            patterns["employee_count"].append(employees_in_bin)
            patterns["customer_count"].append(customers_in_bin)
            patterns["total_count"].append(employees_in_bin + customers_in_bin)

        return patterns

    def operational_insights(self) -> Dict:
        """Generate actionable operational insights"""
        emp_metrics = self.employee_performance_metrics()
        cust_metrics = self.customer_analytics()

        insights = {
            "staff_coverage": {
                "employees_detected": emp_metrics["total_employees"],
                "avg_counter_presence": emp_metrics.get("counter_utilization", 0),
                "recommendation": "",
            },
            "customer_service": {
                "avg_wait_indicator": cust_metrics.get("counter_interaction", 0),
                "customer_to_employee_ratio": cust_metrics["total_customers"]
                / max(emp_metrics["total_employees"], 1),
                "recommendation": "",
            },
            "store_efficiency": {
                "customer_flow": cust_metrics.get("path_directness", 0),
                "browsing_engagement": cust_metrics.get("avg_browsing_time", 0),
                "recommendation": "",
            },
        }

        if emp_metrics.get("counter_utilization", 0) < 50:
            insights["staff_coverage"][
                "recommendation"
            ] = "Counter underutilized. Consider staff reallocation."
        elif emp_metrics.get("counter_utilization", 0) > 90:
            insights["staff_coverage"][
                "recommendation"
            ] = "High counter occupancy. Service capacity optimal."
        else:
            insights["staff_coverage"]["recommendation"] = "Counter coverage looks balanced."

        if insights["customer_service"]["customer_to_employee_ratio"] > 5:
            insights["customer_service"][
                "recommendation"
            ] = "High customer-to-staff ratio. Consider additional staffing."
        elif insights["customer_service"]["customer_to_employee_ratio"] < 2:
            insights["customer_service"][
                "recommendation"
            ] = "Low customer traffic. Staff may be over-allocated."
        else:
            insights["customer_service"]["recommendation"] = "Customer-to-staff ratio looks normal."

        if cust_metrics.get("path_directness", 0) > 0.7:
            insights["store_efficiency"][
                "recommendation"
            ] = "Customers have direct paths. Good for quick service."
        else:
            insights["store_efficiency"][
                "recommendation"
            ] = "Customers browsing extensively. Good for discovery."

        return insights

    def generate_report(self) -> Dict:
        """Generate comprehensive analytics report"""
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "video_duration_sec": self.video_duration,
                "total_tracks": len(self.df),
                "analysis_period": f"{self.video_duration/60:.1f} minutes",
            },
            "employee_metrics": self.employee_performance_metrics(),
            "customer_metrics": self.customer_analytics(),
            "zone_utilization": self.zone_utilization(),
            "traffic_patterns": self.traffic_patterns(),
            "operational_insights": self.operational_insights(),
            "summary": self._generate_summary(),
        }

        return report

    def _generate_summary(self) -> Dict:
        """Generate executive summary"""
        emp_count = len(self.df[self.df["class"] == "Employee"])
        cust_count = len(self.df[self.df["class"] == "Customer"])
        unknown_count = len(self.df[self.df["class"] == "Unknown"])
        total = len(self.df)

        return {
            "total_people_detected": total,
            "employees": emp_count,
            "customers": cust_count,
            "unknown": unknown_count,
            "employee_percentage": (emp_count / total * 100) if total > 0 else 0,
            "customer_percentage": (cust_count / total * 100) if total > 0 else 0,
            "classification_accuracy": ((emp_count + cust_count) / total * 100)
            if total > 0
            else 0,
            "avg_customer_duration": self.df[self.df["class"] == "Customer"][
                "duration_sec"
            ].mean()
            if cust_count > 0
            else 0,
            "avg_employee_duration": self.df[self.df["class"] == "Employee"][
                "duration_sec"
            ].mean()
            if emp_count > 0
            else 0,
        }

    def save_report(self, output_path: str):
        """Save analytics report to JSON"""
        report = self.generate_report()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        return output_path

    def print_summary(self):
        """Print formatted summary to console"""
        report = self.generate_report()

        print("\n" + "=" * 80)
        print("📊 RETAIL ANALYTICS DASHBOARD")
        print("=" * 80)

        summary = report["summary"]
        print("\n📈 SUMMARY")
        print("-" * 80)
        print(f"Analysis Period      : {report['metadata']['analysis_period']}")
        print(f"Total People Detected: {summary['total_people_detected']}")
        print(
            f"  ├─ Employees       : {summary['employees']} ({summary['employee_percentage']:.1f}%)"
        )
        print(
            f"  ├─ Customers       : {summary['customers']} ({summary['customer_percentage']:.1f}%)"
        )
        print(f"  └─ Unknown         : {summary['unknown']}")
        print(f"Classification Rate  : {summary['classification_accuracy']:.1f}%")

        print("\n👥 EMPLOYEE PERFORMANCE")
        print("-" * 80)
        emp = report["employee_metrics"]
        print(f"Active Employees     : {emp['total_employees']}")
        print(f"Avg Work Duration    : {emp['avg_shift_duration']/60:.1f} minutes")
        print(f"Counter Utilization  : {emp['counter_utilization']:.1f}%")
        print(f"Behind Counter Time  : {emp['behind_counter_time']:.1f}%")
        print(f"Avg Mobility         : {emp['mobility_score']:.2f} px/frame")

        print("\n🛒 CUSTOMER ANALYTICS")
        print("-" * 80)
        cust = report["customer_metrics"]
        print(f"Total Customers      : {cust['total_customers']}")
        print(f"Avg Visit Duration   : {cust['avg_visit_duration']:.1f} seconds")
        print(f"Median Visit Duration: {cust['median_visit_duration']:.1f} seconds")
        print(f"Avg Browsing Time    : {cust['avg_browsing_time']:.1f} seconds")
        print(
            f"Path Directness      : {cust['path_directness']:.2f} (0=meandering, 1=direct)"
        )

        print("\nVisit Distribution:")
        print(f"  ├─ Quick (<10s)    : {cust['quick_visits']} customers")
        print(f"  ├─ Short (10-30s)  : {cust['short_visits']} customers")
        print(f"  ├─ Medium (30-60s) : {cust['medium_visits']} customers")
        print(f"  └─ Long (>60s)     : {cust['long_visits']} customers")

        print("\n💡 OPERATIONAL INSIGHTS")
        print("-" * 80)
        insights = report["operational_insights"]

        print("\nStaff Coverage:")
        print(f"  {insights['staff_coverage']['recommendation']}")

        print("\nCustomer Service:")
        print(
            f"  Customer-to-Employee Ratio: {insights['customer_service']['customer_to_employee_ratio']:.1f}:1"
        )
        print(f"  {insights['customer_service']['recommendation']}")

        print("\nStore Efficiency:")
        print(f"  {insights['store_efficiency']['recommendation']}")

        print("\n" + "=" * 80 + "\n")


def generate_html_dashboard(json_report_path: str, output_html: str):
    """Generate interactive HTML dashboard from JSON report"""

    with open(json_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}

        .header h1 {{
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #666;
            font-size: 14px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
        }}

        .stat-card .icon {{
            font-size: 40px;
            margin-bottom: 10px;
        }}

        .stat-card .label {{
            color: #888;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}

        .stat-card .value {{
            color: #333;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-card .subtext {{
            color: #666;
            font-size: 12px;
        }}

        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}

        .chart-container h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 20px;
        }}

        .insights {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .insight-item {{
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            border-radius: 5px;
        }}

        .insight-item h3 {{
            color: #333;
            font-size: 16px;
            margin-bottom: 8px;
        }}

        .insight-item p {{
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }}

        .metric-highlight {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🏪 Retail Analytics Dashboard</h1>
            <p class="subtitle">Generated on {report['metadata']['generated_at']} | Analysis Period: {report['metadata']['analysis_period']}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="icon">👥</div>
                <div class="label">Total People</div>
                <div class="value">{report['summary']['total_people_detected']}</div>
                <div class="subtext">{report['summary']['classification_accuracy']:.1f}% classified</div>
            </div>

            <div class="stat-card">
                <div class="icon">👔</div>
                <div class="label">Employees</div>
                <div class="value">{report['summary']['employees']}</div>
                <div class="subtext">{report['summary']['employee_percentage']:.1f}% of total</div>
            </div>

            <div class="stat-card">
                <div class="icon">🛒</div>
                <div class="label">Customers</div>
                <div class="value">{report['summary']['customers']}</div>
                <div class="subtext">{report['summary']['customer_percentage']:.1f}% of total</div>
            </div>

            <div class="stat-card">
                <div class="icon">⏱️</div>
                <div class="label">Avg Customer Visit</div>
                <div class="value">{report['summary']['avg_customer_duration']:.0f}s</div>
                <div class="subtext">Average duration</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>📊 Classification Distribution</h2>
            <canvas id="classChart" height="100"></canvas>
        </div>

        <div class="chart-container">
            <h2>🎯 Zone Utilization</h2>
            <canvas id="zoneChart" height="100"></canvas>
        </div>

        <div class="chart-container">
            <h2>🕒 Customer Visit Duration Distribution</h2>
            <canvas id="visitChart" height="100"></canvas>
        </div>

        <div class="insights">
            <h2 style="margin-bottom: 20px; color: #333;">💡 Operational Insights</h2>

            <div class="insight-item">
                <h3>Staff Coverage</h3>
                <p>
                    <span class="metric-highlight">{report['employee_metrics']['total_employees']} employees</span> detected with
                    <span class="metric-highlight">{report['employee_metrics']['counter_utilization']:.1f}%</span> counter utilization.
                </p>
                <p style="margin-top: 8px; font-style: italic;">
                    💭 {report['operational_insights']['staff_coverage']['recommendation']}
                </p>
            </div>

            <div class="insight-item">
                <h3>Customer Service</h3>
                <p>
                    Customer-to-employee ratio: <span class="metric-highlight">{report['operational_insights']['customer_service']['customer_to_employee_ratio']:.1f}:1</span>
                </p>
                <p style="margin-top: 8px; font-style: italic;">
                    💭 {report['operational_insights']['customer_service']['recommendation']}
                </p>
            </div>

            <div class="insight-item">
                <h3>Store Efficiency</h3>
                <p>
                    Customer path directness: <span class="metric-highlight">{report['customer_metrics']['path_directness']:.2f}</span> |
                    Avg browsing time: <span class="metric-highlight">{report['customer_metrics']['avg_browsing_time']:.0f}s</span>
                </p>
                <p style="margin-top: 8px; font-style: italic;">
                    💭 {report['operational_insights']['store_efficiency']['recommendation']}
                </p>
            </div>
        </div>
    </div>

    <script>
        // Classification Distribution Chart
        new Chart(document.getElementById('classChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Employees', 'Customers', 'Unknown'],
                datasets: [{{
                    data: [{report['summary']['employees']}, {report['summary']['customers']}, {report['summary']['unknown']}],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});

        // Zone Utilization Chart
        const zoneData = {json.dumps(report['zone_utilization'], ensure_ascii=False)};
        new Chart(document.getElementById('zoneChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(zoneData).map(z => z.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                datasets: [
                    {{
                        label: 'Employee Time (s)',
                        data: Object.values(zoneData).map(z => z.employee_time_sec)
                    }},
                    {{
                        label: 'Customer Time (s)',
                        data: Object.values(zoneData).map(z => z.customer_time_sec)
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        stacked: true
                    }},
                    y: {{
                        stacked: true,
                        beginAtZero: true
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});

        // Visit Duration Distribution
        new Chart(document.getElementById('visitChart'), {{
            type: 'bar',
            data: {{
                labels: ['Quick (<10s)', 'Short (10-30s)', 'Medium (30-60s)', 'Long (>60s)'],
                datasets: [{{
                    label: 'Number of Customers',
                    data: [
                        {report['customer_metrics']['quick_visits']},
                        {report['customer_metrics']['short_visits']},
                        {report['customer_metrics']['medium_visits']},
                        {report['customer_metrics']['long_visits']}
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    # ✅ IMPORTANT FIX: write as UTF-8 so emojis won't crash on Windows
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate retail analytics dashboard")
    parser.add_argument("--results", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")

    args = parser.parse_args()

    print("\n📊 Generating analytics...")
    analytics = RetailAnalytics(args.results, args.fps)

    analytics.print_summary()

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, "analytics_report.json")
    analytics.save_report(json_path)
    print(f"✅ JSON report saved: {json_path}")

    html_path = os.path.join(args.output_dir, "dashboard.html")
    generate_html_dashboard(json_path, html_path)
    print(f"✅ HTML dashboard saved: {html_path}")
    print(f"\n🌐 Open {html_path} in your browser to view the interactive dashboard!")


if __name__ == "__main__":
    main()
