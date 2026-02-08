"""
Chart rendering tools for AI assistant.
Provides function definitions and execution for chart generation.
"""
import json
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from enum import Enum


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"


# Function definitions for LLM
CHART_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "render_chart",
            "description": "Render a chart visualization based on data. Use this when the user wants to visualize data, create graphs, or display statistical information. Supports bar charts, line charts, pie charts, scatter plots, area charts, and histograms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "pie", "scatter", "area", "histogram"],
                        "description": "The type of chart to render"
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title"
                    },
                    "data": {
                        "type": "object",
                        "description": "Chart data with 'labels' (list of strings for x-axis or categories) and 'values' (list of numbers)",
                        "properties": {
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Labels for categories or x-axis points"
                            },
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Values to plot"
                            },
                            "series_name": {
                                "type": "string",
                                "description": "Name of the data series (optional, for line/area charts)"
                            }
                        },
                        "required": ["labels", "values"]
                    },
                    "x_label": {
                        "type": "string",
                        "description": "Label for x-axis (optional)"
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Label for y-axis (optional)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional description or notes about the chart"
                    }
                },
                "required": ["chart_type", "title", "data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "render_comparison_chart",
            "description": "Render a comparison chart with multiple series. Use this when comparing multiple datasets or showing trends across different categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "area"],
                        "description": "The type of chart to render"
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categories or x-axis labels"
                    },
                    "series": {
                        "type": "array",
                        "description": "Multiple data series to compare",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Series name"},
                                "values": {"type": "array", "items": {"type": "number"}, "description": "Values for this series"}
                            },
                            "required": ["name", "values"]
                        }
                    },
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["chart_type", "title", "categories", "series"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "render_data_table",
            "description": "Render a formatted data table. Use this when the user wants to see data in tabular format or display structured information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Table title"
                    },
                    "headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column headers"
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "Table rows, each row is a list of values"
                    },
                    "description": {"type": "string"}
                },
                "required": ["title", "headers", "rows"]
            }
        }
    }
]




def execute_chart_function(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a chart function with the given arguments.

    Args:
        function_name: Name of the function to execute
        arguments: Function arguments parsed from JSON

    Returns:
        Dict with 'success' status and optional 'error' message
    """
    try:
        if function_name == "render_chart":
            return render_chart(**arguments)
        elif function_name == "render_comparison_chart":
            return render_comparison_chart(**arguments)
        elif function_name == "render_data_table":
            return render_data_table(**arguments)
        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_chart(
    chart_type: str,
    title: str,
    data: Dict[str, Any],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Render a single-series chart."""
    try:
        chart_type_enum = ChartType(chart_type.lower())

        # Create DataFrame
        df = pd.DataFrame({
            'labels': data['labels'],
            'values': data['values']
        })

        st.markdown(f"### {title}")

        if description:
            st.markdown(f"*{description}*")

        # Render based on chart type
        if chart_type_enum == ChartType.BAR:
            st.bar_chart(data=df.set_index('labels')['values'], use_container_width=True)

        elif chart_type_enum == ChartType.LINE:
            st.line_chart(data=df.set_index('labels')['values'], use_container_width=True)

        elif chart_type_enum == ChartType.AREA:
            st.area_chart(data=df.set_index('labels')['values'], use_container_width=True)

        elif chart_type_enum == ChartType.PIE:
            # For pie chart, use matplotlib for better visuals
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(df['values'], labels=df['labels'], autopct='%1.1f%%', startangle=90)
            ax.set_title(title)
            st.pyplot(fig)

        elif chart_type_enum == ChartType.SCATTER:
            st.scatter_chart(data=df, x='labels', y='values')

        elif chart_type_enum == ChartType.HISTOGRAM:
            st.bar_chart(data=df.set_index('labels')['values'], use_container_width=True)

        # Display axis labels if provided
        if x_label or y_label:
            cols = st.columns(2)
            with cols[0]:
                if x_label:
                    st.caption(f"X轴: {x_label}")
            with cols[1]:
                if y_label:
                    st.caption(f"Y轴: {y_label}")

        return {"success": True}

    except Exception as e:
        st.error(f"图表渲染失败: {str(e)}")
        return {"success": False, "error": str(e)}


def render_comparison_chart(
    chart_type: str,
    title: str,
    categories: List[str],
    series: List[Dict[str, Any]],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Render a multi-series comparison chart."""
    try:
        st.markdown(f"### {title}")

        if description:
            st.markdown(f"*{description}*")

        # Create DataFrame with multiple series
        data_dict = {'category': categories}
        for s in series:
            data_dict[s['name']] = s['values']

        df = pd.DataFrame(data_dict).set_index('category')

        chart_type_enum = ChartType(chart_type.lower())

        # Render based on chart type
        if chart_type_enum == ChartType.BAR:
            st.bar_chart(data=df, use_container_width=True)
        elif chart_type_enum in [ChartType.LINE, ChartType.AREA]:
            st.line_chart(data=df, use_container_width=True)
        else:
            st.bar_chart(data=df, use_container_width=True)

        # Display axis labels
        if x_label or y_label:
            cols = st.columns(2)
            with cols[0]:
                if x_label:
                    st.caption(f"X轴: {x_label}")
            with cols[1]:
                if y_label:
                    st.caption(f"Y轴: {y_label}")

        return {"success": True}

    except Exception as e:
        st.error(f"对比图表渲染失败: {str(e)}")
        return {"success": False, "error": str(e)}


def render_data_table(
    title: str,
    headers: List[str],
    rows: List[List[str]],
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Render a formatted data table."""
    try:
        st.markdown(f"### {title}")

        if description:
            st.markdown(f"*{description}*")

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Display as styled table
        st.dataframe(df, use_container_width=True, hide_index=True)

        return {"success": True}

    except Exception as e:
        st.error(f"表格渲染失败: {str(e)}")
        return {"success": False, "error": str(e)}


CHART_TOOL_HANDLERS = {
    "render_chart": render_chart,
    "render_comparison_chart": render_comparison_chart,
    "render_data_table": render_data_table,
}


def should_use_chart_tools(prompt: str) -> bool:
    """Return True if the prompt likely needs chart tooling."""
    if not prompt:
        return False
    keywords = [
        "chart", "graph", "visualize", "plot",
        "图表", "图", "可视化", "绘制"
    ]
    lowered = prompt.lower()
    return any(keyword in lowered for keyword in keywords)


# System prompt for chart assistant
CHART_ASSISTANT_PROMPT = """You are a Chart AI Assistant with data visualization capabilities.

When users ask for charts, graphs, or data visualization:
1. Analyze the data they provide or generate appropriate sample data
2. Use the available chart functions to create visualizations
3. Provide insights about the data along with the visualization

Available chart types:
- bar: Good for comparing categories
- line: Good for showing trends over time
- pie: Good for showing proportions/percentages
- scatter: Good for showing relationships between variables
- area: Good for showing cumulative trends
- histogram: Good for showing distribution

Always ensure data labels and values are properly formatted as lists of strings and numbers respectively.
"""
