"""
Real-time performance dashboard for the order book simulator.

Provides interactive visualizations of throughput, latency, order book depth,
and system performance metrics using Plotly and Dash.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional
from collections import deque
import json

from profiling.performance_monitor import get_performance_monitor
from core.orderbook import OrderBookManager
from simulation.simulation_engine import SimulationEngine


class DashboardData:
    """Thread-safe data container for dashboard metrics."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.lock = threading.Lock()
        
        # Time series data
        self.timestamps = deque(maxlen=max_points)
        self.throughput_data = deque(maxlen=max_points)
        self.latency_data = deque(maxlen=max_points)
        self.cpu_data = deque(maxlen=max_points)
        self.memory_data = deque(maxlen=max_points)
        self.trade_volume_data = deque(maxlen=max_points)
        
        # Order book data
        self.bid_prices = []
        self.bid_quantities = []
        self.ask_prices = []
        self.ask_quantities = []
        
        # Summary statistics
        self.summary_stats = {}
        
        # Last update time
        self.last_update = time.time()
    
    def update_metrics(self, performance_data: Dict[str, Any]):
        """Update dashboard data with new performance metrics."""
        with self.lock:
            current_time = time.time()
            self.timestamps.append(current_time)
            
            # Extract metrics
            perf_metrics = performance_data.get('performance_metrics', {})
            system_resources = performance_data.get('system_resources', {})
            session_metrics = performance_data.get('session_metrics', {})
            
            # Throughput
            throughput_stats = perf_metrics.get('throughput_ops_per_sec', {})
            self.throughput_data.append(throughput_stats.get('current', 0))
            
            # Latency
            latency_stats = perf_metrics.get('order_processing_latency_ms', {})
            self.latency_data.append(latency_stats.get('mean', 0))
            
            # System resources
            self.cpu_data.append(system_resources.get('process_cpu_percent', 0))
            self.memory_data.append(system_resources.get('process_memory_mb', 0))
            
            # Trade volume
            self.trade_volume_data.append(session_metrics.get('total_volume', 0))
            
            # Update summary
            self.summary_stats = {
                'total_orders': session_metrics.get('orders_processed', 0),
                'total_trades': session_metrics.get('trades_executed', 0),
                'total_volume': session_metrics.get('total_volume', 0),
                'peak_throughput': session_metrics.get('peak_throughput_ops_per_sec', 0),
                'current_throughput': throughput_stats.get('current', 0),
                'avg_latency': latency_stats.get('mean', 0),
                'p99_latency': latency_stats.get('p99', 0),
                'cpu_usage': system_resources.get('process_cpu_percent', 0),
                'memory_usage': system_resources.get('process_memory_mb', 0),
            }
            
            self.last_update = current_time
    
    def update_order_book(self, order_book_data: Dict[str, Any]):
        """Update order book depth data."""
        with self.lock:
            # Extract bid/ask data
            bids = order_book_data.get('bids', [])
            asks = order_book_data.get('asks', [])
            
            self.bid_prices = [level['price'] for level in bids[:10]]  # Top 10 levels
            self.bid_quantities = [level['quantity'] for level in bids[:10]]
            self.ask_prices = [level['price'] for level in asks[:10]]
            self.ask_quantities = [level['quantity'] for level in asks[:10]]
    
    def get_time_series_data(self) -> Dict[str, List]:
        """Get time series data for plotting."""
        with self.lock:
            return {
                'timestamps': list(self.timestamps),
                'throughput': list(self.throughput_data),
                'latency': list(self.latency_data),
                'cpu': list(self.cpu_data),
                'memory': list(self.memory_data),
                'volume': list(self.trade_volume_data),
            }
    
    def get_order_book_data(self) -> Dict[str, List]:
        """Get order book data for plotting."""
        with self.lock:
            return {
                'bid_prices': list(self.bid_prices),
                'bid_quantities': list(self.bid_quantities),
                'ask_prices': list(self.ask_prices),
                'ask_quantities': list(self.ask_quantities),
            }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self.lock:
            return dict(self.summary_stats)


class PerformanceDashboard:
    """
    Interactive performance dashboard for order book monitoring.
    
    Provides real-time visualization of system performance, order book depth,
    and trading activity with automatic updates.
    """
    
    def __init__(self, update_interval: int = 1000, port: int = 8050):
        self.update_interval = update_interval  # milliseconds
        self.port = port
        
        # Data container
        self.dashboard_data = DashboardData()
        
        # Performance monitor
        self.performance_monitor = get_performance_monitor()
        
        # Data update thread
        self.update_thread = None
        self.is_updating = False
        
        # Dash app
        self.app = self._create_app()
        
        # Order book manager (optional)
        self.order_book_manager = None
    
    def set_order_book_manager(self, manager: OrderBookManager):
        """Set order book manager for depth visualization."""
        self.order_book_manager = manager
    
    def _create_app(self) -> dash.Dash:
        """Create the Dash application."""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Order Book Performance Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Hr(),
            ]),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H3("Total Orders", style={'textAlign': 'center'}),
                    html.H2(id='total-orders', style={'textAlign': 'center', 'color': '#3498db'}),
                ], className='summary-card', style={'width': '15%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.H3("Total Trades", style={'textAlign': 'center'}),
                    html.H2(id='total-trades', style={'textAlign': 'center', 'color': '#e74c3c'}),
                ], className='summary-card', style={'width': '15%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.H3("Throughput", style={'textAlign': 'center'}),
                    html.H2(id='current-throughput', style={'textAlign': 'center', 'color': '#2ecc71'}),
                ], className='summary-card', style={'width': '15%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.H3("Avg Latency", style={'textAlign': 'center'}),
                    html.H2(id='avg-latency', style={'textAlign': 'center', 'color': '#f39c12'}),
                ], className='summary-card', style={'width': '15%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.H3("CPU Usage", style={'textAlign': 'center'}),
                    html.H2(id='cpu-usage', style={'textAlign': 'center', 'color': '#9b59b6'}),
                ], className='summary-card', style={'width': '15%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.H3("Memory", style={'textAlign': 'center'}),
                    html.H2(id='memory-usage', style={'textAlign': 'center', 'color': '#1abc9c'}),
                ], className='summary-card', style={'width': '15%', 'display': 'inline-block', 'margin': '10px'}),
            ], style={'textAlign': 'center'}),
            
            html.Hr(),
            
            # Charts row 1
            html.Div([
                html.Div([
                    dcc.Graph(id='throughput-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='latency-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            
            # Charts row 2
            html.Div([
                html.Div([
                    dcc.Graph(id='system-resources-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='order-book-depth-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
            
            # Store for data
            dcc.Store(id='dashboard-data-store'),
        ])
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _register_callbacks(self, app: dash.Dash):
        """Register Dash callbacks for interactivity."""
        
        @app.callback(
            [Output('total-orders', 'children'),
             Output('total-trades', 'children'),
             Output('current-throughput', 'children'),
             Output('avg-latency', 'children'),
             Output('cpu-usage', 'children'),
             Output('memory-usage', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_summary_cards(n):
            stats = self.dashboard_data.get_summary_stats()
            
            return (
                f"{stats.get('total_orders', 0):,}",
                f"{stats.get('total_trades', 0):,}",
                f"{stats.get('current_throughput', 0):.0f}/s",
                f"{stats.get('avg_latency', 0):.2f}ms",
                f"{stats.get('cpu_usage', 0):.1f}%",
                f"{stats.get('memory_usage', 0):.0f}MB",
            )
        
        @app.callback(
            Output('throughput-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_throughput_chart(n):
            data = self.dashboard_data.get_time_series_data()
            
            if not data['timestamps']:
                return go.Figure()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['throughput'],
                mode='lines',
                name='Throughput',
                line=dict(color='#3498db', width=2)
            ))
            
            fig.update_layout(
                title='Order Processing Throughput',
                xaxis_title='Time',
                yaxis_title='Orders/Second',
                template='plotly_white',
                height=400
            )
            
            return fig
        
        @app.callback(
            Output('latency-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_latency_chart(n):
            data = self.dashboard_data.get_time_series_data()
            
            if not data['timestamps']:
                return go.Figure()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['latency'],
                mode='lines',
                name='Avg Latency',
                line=dict(color='#e74c3c', width=2)
            ))
            
            fig.update_layout(
                title='Order Processing Latency',
                xaxis_title='Time',
                yaxis_title='Latency (ms)',
                template='plotly_white',
                height=400
            )
            
            return fig
        
        @app.callback(
            Output('system-resources-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_system_resources_chart(n):
            data = self.dashboard_data.get_time_series_data()
            
            if not data['timestamps']:
                return go.Figure()
            
            fig = go.Figure()
            
            # CPU usage
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['cpu'],
                mode='lines',
                name='CPU %',
                line=dict(color='#9b59b6', width=2),
                yaxis='y'
            ))
            
            # Memory usage (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['memory'],
                mode='lines',
                name='Memory MB',
                line=dict(color='#1abc9c', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='System Resource Usage',
                xaxis_title='Time',
                yaxis=dict(title='CPU Usage (%)', side='left'),
                yaxis2=dict(title='Memory Usage (MB)', side='right', overlaying='y'),
                template='plotly_white',
                height=400
            )
            
            return fig
        
        @app.callback(
            Output('order-book-depth-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_order_book_chart(n):
            ob_data = self.dashboard_data.get_order_book_data()
            
            if not ob_data['bid_prices'] and not ob_data['ask_prices']:
                return go.Figure().add_annotation(
                    text="No order book data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            fig = go.Figure()
            
            # Bids
            if ob_data['bid_prices']:
                fig.add_trace(go.Bar(
                    x=ob_data['bid_quantities'],
                    y=ob_data['bid_prices'],
                    orientation='h',
                    name='Bids',
                    marker_color='#2ecc71',
                    opacity=0.7
                ))
            
            # Asks
            if ob_data['ask_prices']:
                fig.add_trace(go.Bar(
                    x=[-q for q in ob_data['ask_quantities']],  # Negative for left side
                    y=ob_data['ask_prices'],
                    orientation='h',
                    name='Asks',
                    marker_color='#e74c3c',
                    opacity=0.7
                ))
            
            fig.update_layout(
                title='Order Book Depth',
                xaxis_title='Quantity',
                yaxis_title='Price',
                template='plotly_white',
                height=400,
                barmode='overlay'
            )
            
            return fig
    
    def start_data_updates(self):
        """Start automatic data updates."""
        if self.is_updating:
            return
        
        self.is_updating = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        print("Dashboard data updates started")
    
    def stop_data_updates(self):
        """Stop automatic data updates."""
        self.is_updating = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        print("Dashboard data updates stopped")
    
    def _update_loop(self):
        """Data update loop."""
        while self.is_updating:
            try:
                # Get performance data
                performance_data = self.performance_monitor.get_performance_summary()
                self.dashboard_data.update_metrics(performance_data)
                
                # Get order book data if available
                if self.order_book_manager:
                    # Get first order book for depth visualization
                    order_books = self.order_book_manager.get_all_order_books()
                    if order_books:
                        first_symbol = list(order_books.keys())[0]
                        order_book = order_books[first_symbol]
                        snapshot = order_book.get_market_data_snapshot()
                        
                        # Convert to dashboard format
                        ob_data = {
                            'bids': [{'price': level.price, 'quantity': level.total_quantity} 
                                   for level in snapshot.bid_levels],
                            'asks': [{'price': level.price, 'quantity': level.total_quantity} 
                                   for level in snapshot.ask_levels],
                        }
                        self.dashboard_data.update_order_book(ob_data)
                
                time.sleep(self.update_interval / 1000.0)  # Convert ms to seconds
                
            except Exception as e:
                print(f"Error in dashboard update loop: {e}")
                time.sleep(1.0)
    
    def run(self, debug: bool = False, host: str = '127.0.0.1'):
        """Run the dashboard server."""
        print(f"Starting dashboard server on http://{host}:{self.port}")
        
        # Start data updates
        self.start_data_updates()
        
        try:
            self.app.run_server(debug=debug, host=host, port=self.port)
        finally:
            self.stop_data_updates()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_data_updates()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_data_updates()


def create_dashboard(port: int = 8050, update_interval: int = 1000) -> PerformanceDashboard:
    """
    Create a configured performance dashboard.
    
    Args:
        port: Server port
        update_interval: Update interval in milliseconds
    
    Returns:
        Configured PerformanceDashboard instance
    """
    return PerformanceDashboard(update_interval=update_interval, port=port)
