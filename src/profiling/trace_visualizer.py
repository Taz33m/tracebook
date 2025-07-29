"""
Trace visualization tools for performance analysis.

Provides interactive charts and reports for trace data analysis,
similar to magic-trace's visualization capabilities.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


class TraceVisualizer:
    """
    Interactive trace data visualizer.
    
    Creates detailed charts and reports from trace analysis data.
    """
    
    def __init__(self, trace_data: Dict[str, Any] = None):
        self.trace_data = trace_data
        self.figures = {}
    
    def load_trace_data(self, file_path: str) -> bool:
        """Load trace data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                self.trace_data = json.load(f)
            return True
        except Exception as e:
            print(f"Failed to load trace data: {e}")
            return False
    
    def create_function_timeline(self) -> go.Figure:
        """Create a timeline chart of function calls."""
        if not self.trace_data or 'completed_calls' not in self.trace_data:
            return None
        
        calls = self.trace_data['completed_calls']
        
        # Prepare data for timeline
        timeline_data = []
        for call in calls:
            timeline_data.append({
                'Function': call['function_name'],
                'Start': call['start_time_ns'] / 1_000_000,  # Convert to ms
                'Duration': call['duration_ns'] / 1_000_000,  # Convert to ms
                'End': call['end_time_ns'] / 1_000_000,  # Convert to ms
                'Thread': call['thread_id'],
                'Depth': call['call_depth']
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create Gantt-style chart
        fig = go.Figure()
        
        # Color by function
        unique_functions = df['Function'].unique()
        colors = px.colors.qualitative.Set3[:len(unique_functions)]
        color_map = dict(zip(unique_functions, colors))
        
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Start'], row['End']],
                y=[row['Function'], row['Function']],
                mode='lines',
                line=dict(
                    color=color_map[row['Function']],
                    width=8
                ),
                name=row['Function'],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Function']}</b><br>"
                    f"Duration: {row['Duration']:.3f}ms<br>"
                    f"Start: {row['Start']:.3f}ms<br>"
                    f"Thread: {row['Thread']}<br>"
                    f"Depth: {row['Depth']}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="Function Call Timeline",
            xaxis_title="Time (ms)",
            yaxis_title="Function",
            height=600,
            showlegend=False
        )
        
        self.figures['timeline'] = fig
        return fig
    
    def create_function_performance_chart(self) -> go.Figure:
        """Create performance comparison chart for functions."""
        if not self.trace_data or 'function_analysis' not in self.trace_data:
            return None
        
        analysis = self.trace_data['function_analysis']
        
        # Prepare data
        functions = list(analysis.keys())
        mean_durations = [analysis[f]['mean_duration_ms'] for f in functions]
        total_times = [analysis[f]['total_time_ms'] for f in functions]
        call_counts = [analysis[f]['call_count'] for f in functions]
        p99_durations = [analysis[f]['p99_duration_ns'] / 1_000_000 for f in functions]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Duration per Call',
                'Total Time Consumed',
                'Call Count',
                'P99 Latency'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Mean duration
        fig.add_trace(
            go.Bar(x=functions, y=mean_durations, name="Mean Duration (ms)"),
            row=1, col=1
        )
        
        # Total time
        fig.add_trace(
            go.Bar(x=functions, y=total_times, name="Total Time (ms)"),
            row=1, col=2
        )
        
        # Call count
        fig.add_trace(
            go.Bar(x=functions, y=call_counts, name="Call Count"),
            row=2, col=1
        )
        
        # P99 latency
        fig.add_trace(
            go.Bar(x=functions, y=p99_durations, name="P99 Latency (ms)"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Function Performance Analysis",
            height=800,
            showlegend=False
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        self.figures['performance'] = fig
        return fig
    
    def create_latency_distribution(self) -> go.Figure:
        """Create latency distribution charts."""
        if not self.trace_data or 'completed_calls' not in self.trace_data:
            return None
        
        calls = self.trace_data['completed_calls']
        
        # Group by function
        function_latencies = {}
        for call in calls:
            func_name = call['function_name']
            latency_ms = call['duration_ns'] / 1_000_000
            
            if func_name not in function_latencies:
                function_latencies[func_name] = []
            function_latencies[func_name].append(latency_ms)
        
        # Create distribution plots
        fig = go.Figure()
        
        for func_name, latencies in function_latencies.items():
            fig.add_trace(go.Histogram(
                x=latencies,
                name=func_name,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Latency Distribution by Function",
            xaxis_title="Latency (ms)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=500
        )
        
        self.figures['distribution'] = fig
        return fig
    
    def create_call_depth_analysis(self) -> go.Figure:
        """Create call depth analysis chart."""
        if not self.trace_data or 'completed_calls' not in self.trace_data:
            return None
        
        calls = self.trace_data['completed_calls']
        
        # Analyze call depths
        depth_data = {}
        for call in calls:
            depth = call['call_depth']
            func_name = call['function_name']
            
            if depth not in depth_data:
                depth_data[depth] = {}
            if func_name not in depth_data[depth]:
                depth_data[depth][func_name] = 0
            depth_data[depth][func_name] += 1
        
        # Create stacked bar chart
        fig = go.Figure()
        
        depths = sorted(depth_data.keys())
        all_functions = set()
        for depth_funcs in depth_data.values():
            all_functions.update(depth_funcs.keys())
        
        colors = px.colors.qualitative.Set3[:len(all_functions)]
        color_map = dict(zip(all_functions, colors))
        
        for func_name in all_functions:
            counts = [depth_data.get(depth, {}).get(func_name, 0) for depth in depths]
            fig.add_trace(go.Bar(
                x=depths,
                y=counts,
                name=func_name,
                marker_color=color_map[func_name]
            ))
        
        fig.update_layout(
            title="Function Calls by Call Depth",
            xaxis_title="Call Depth",
            yaxis_title="Number of Calls",
            barmode='stack',
            height=500
        )
        
        self.figures['call_depth'] = fig
        return fig
    
    def create_performance_summary(self) -> go.Figure:
        """Create overall performance summary."""
        if not self.trace_data or 'summary' not in self.trace_data:
            return None
        
        summary = self.trace_data['summary']
        
        # Create metrics table
        metrics = [
            ("Total Function Calls", summary.get('total_function_calls', 0)),
            ("Unique Functions", summary.get('unique_functions', 0)),
            ("Total Traced Time", f"{summary.get('total_traced_time_ms', 0):.2f} ms"),
            ("Trace Overhead", f"{summary.get('trace_overhead_percentage', 0):.2f}%"),
            ("Max Call Depth", summary.get('max_call_depth', 0)),
            ("Active Threads", summary.get('active_threads', 0)),
        ]
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='lightblue',
                align='left',
                font=dict(size=14, color='black')
            ),
            cells=dict(
                values=[[m[0] for m in metrics], [m[1] for m in metrics]],
                fill_color='white',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="Performance Summary",
            height=300
        )
        
        self.figures['summary'] = fig
        return fig
    
    def create_insights_report(self) -> str:
        """Create a text-based insights report."""
        if not self.trace_data:
            return "No trace data available"
        
        report = []
        report.append("PERFORMANCE INSIGHTS REPORT")
        report.append("=" * 50)
        
        # Summary section
        if 'summary' in self.trace_data:
            summary = self.trace_data['summary']
            report.append("\nSUMMARY:")
            report.append(f"  Total function calls: {summary.get('total_function_calls', 0):,}")
            report.append(f"  Unique functions: {summary.get('unique_functions', 0)}")
            report.append(f"  Total traced time: {summary.get('total_traced_time_ms', 0):.2f} ms")
            report.append(f"  Trace overhead: {summary.get('trace_overhead_percentage', 0):.2f}%")
        
        # Function analysis
        if 'function_analysis' in self.trace_data:
            analysis = self.trace_data['function_analysis']
            
            # Top functions by total time
            sorted_by_total = sorted(
                analysis.items(),
                key=lambda x: x[1]['total_time_ms'],
                reverse=True
            )
            
            report.append("\nTOP FUNCTIONS BY TOTAL TIME:")
            for i, (func_name, stats) in enumerate(sorted_by_total[:5]):
                report.append(
                    f"  {i+1}. {func_name}: {stats['total_time_ms']:.2f}ms "
                    f"({stats['call_count']} calls)"
                )
            
            # Slowest functions by average
            sorted_by_avg = sorted(
                analysis.items(),
                key=lambda x: x[1]['mean_duration_ms'],
                reverse=True
            )
            
            report.append("\nSLOWEST FUNCTIONS BY AVERAGE:")
            for i, (func_name, stats) in enumerate(sorted_by_avg[:5]):
                report.append(
                    f"  {i+1}. {func_name}: {stats['mean_duration_ms']:.3f}ms avg "
                    f"(max: {stats['max_duration_ns']/1_000_000:.3f}ms)"
                )
        
        # Performance insights
        if 'performance_insights' in self.trace_data:
            insights = self.trace_data['performance_insights']
            if insights:
                report.append("\nPERFORMANCE INSIGHTS:")
                for insight in insights:
                    report.append(f"  • {insight}")
        
        return "\n".join(report)
    
    def generate_html_report(self, output_file: str) -> bool:
        """Generate comprehensive HTML report with all visualizations."""
        try:
            # Create all figures
            self.create_function_timeline()
            self.create_function_performance_chart()
            self.create_latency_distribution()
            self.create_call_depth_analysis()
            self.create_performance_summary()
            
            # Generate HTML content
            html_content = []
            html_content.append("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trace Analysis Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .chart-container { margin: 20px 0; }
                    .insights { background: #f5f5f5; padding: 15px; margin: 20px 0; }
                    pre { background: #f0f0f0; padding: 10px; overflow-x: auto; }
                </style>
            </head>
            <body>
                <h1>Trace Analysis Report</h1>
            """)
            
            # Add insights report
            insights_text = self.create_insights_report()
            html_content.append(f"""
                <div class="insights">
                    <h2>Performance Insights</h2>
                    <pre>{insights_text}</pre>
                </div>
            """)
            
            # Add charts
            for chart_name, fig in self.figures.items():
                if fig:
                    chart_html = pyo.plot(fig, output_type='div', include_plotlyjs=False)
                    html_content.append(f"""
                        <div class="chart-container">
                            <h2>{chart_name.replace('_', ' ').title()}</h2>
                            {chart_html}
                        </div>
                    """)
            
            html_content.append("</body></html>")
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write('\n'.join(html_content))
            
            return True
            
        except Exception as e:
            print(f"Failed to generate HTML report: {e}")
            return False
    
    def save_charts(self, output_dir: str) -> bool:
        """Save individual charts as PNG files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create all figures if not already created
            if not self.figures:
                self.create_function_timeline()
                self.create_function_performance_chart()
                self.create_latency_distribution()
                self.create_call_depth_analysis()
                self.create_performance_summary()
            
            # Save each figure
            for chart_name, fig in self.figures.items():
                if fig:
                    output_file = output_path / f"{chart_name}.png"
                    fig.write_image(str(output_file), width=1200, height=800)
            
            return True
            
        except Exception as e:
            print(f"Failed to save charts: {e}")
            return False


def visualize_trace_file(trace_file: str, output_dir: str = "trace_reports") -> bool:
    """
    Convenience function to visualize a trace file.
    
    Args:
        trace_file: Path to JSON trace file
        output_dir: Directory to save visualization outputs
    
    Returns:
        True if successful, False otherwise
    """
    visualizer = TraceVisualizer()
    
    if not visualizer.load_trace_data(trace_file):
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML report
    html_file = output_path / "trace_report.html"
    if not visualizer.generate_html_report(str(html_file)):
        return False
    
    print(f"Trace visualization report generated: {html_file}")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python trace_visualizer.py <trace_file.json>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    if not Path(trace_file).exists():
        print(f"Trace file not found: {trace_file}")
        sys.exit(1)
    
    success = visualize_trace_file(trace_file)
    sys.exit(0 if success else 1)
