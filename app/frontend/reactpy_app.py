from reactpy import component, html, hooks
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define a Button component that demonstrates state and props
@component
def Button(label, on_click, color="blue"):
    # Define CSS styles as Python dictionaries
    button_style = {
        "backgroundColor": color,
        "color": "white",
        "padding": "10px 15px",
        "border": "none",
        "borderRadius": "4px",
        "cursor": "pointer",
        "fontSize": "16px",
        "margin": "5px",
        "transition": "background-color 0.3s ease"
    }
    
    # Hover state management
    is_hovering, set_is_hovering = hooks.use_state(False)
    
    if is_hovering:
        button_style["backgroundColor"] = "darkblue" if color == "blue" else "darkgreen" if color == "green" else "darkred"
    
    return html.button(
        {"style": button_style, 
         "onClick": on_click,
         "onMouseEnter": lambda _: set_is_hovering(True),
         "onMouseLeave": lambda _: set_is_hovering(False)},
        label
    )

# Define a Card component for layout structure
@component
def Card(title, children):
    card_style = {
        "border": "1px solid #ddd",
        "borderRadius": "8px",
        "padding": "20px",
        "margin": "10px 0",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
        "backgroundColor": "white"
    }
    
    title_style = {
        "fontSize": "20px",
        "marginBottom": "15px",
        "color": "#333",
        "borderBottom": "1px solid #eee",
        "paddingBottom": "10px"
    }
    
    return html.div(
        {"style": card_style},
        html.h2({"style": title_style}, title),
        children
    )

# Define a Counter component that demonstrates state management
@component
def Counter():
    count, set_count = hooks.use_state(0)
    
    def increment(_):
        logger.info("Incrementing counter")
        set_count(count + 1)
    
    def decrement(_):
        logger.info("Decrementing counter")
        set_count(count - 1)
    
    return html.div(
        html.h3("Counter Example"),
        html.p(f"Current count: {count}"),
        html.div(
            Button("Increment", increment, "green"),
            Button("Decrement", decrement, "red")
        )
    )

# Define a TodoList component that demonstrates more complex state management
@component
def TodoList():
    todos, set_todos = hooks.use_state([])
    new_todo, set_new_todo = hooks.use_state("")
    
    def add_todo(_):
        if new_todo.strip():
            logger.info(f"Adding todo: {new_todo}")
            set_todos(todos + [new_todo])
            set_new_todo("")
    
    def remove_todo(index):
        logger.info(f"Removing todo at index: {index}")
        new_todos = todos.copy()
        new_todos.pop(index)
        set_todos(new_todos)
    
    input_style = {
        "padding": "8px",
        "fontSize": "16px",
        "borderRadius": "4px",
        "border": "1px solid #ddd",
        "marginRight": "10px",
        "width": "300px"
    }
    
    todo_item_style = {
        "padding": "10px",
        "borderBottom": "1px solid #eee",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center"
    }
    
    return html.div(
        html.h3("Todo List Example"),
        html.div(
            html.input({
                "style": input_style,
                "value": new_todo,
                "onChange": lambda e: set_new_todo(e["target"]["value"]),
                "placeholder": "Add a new task...",
                "onKeyPress": lambda e: add_todo(e) if e["key"] == "Enter" else None
            }),
            Button("Add", add_todo, "blue")
        ),
        html.ul(
            {"style": {"listStyleType": "none", "padding": "0", "marginTop": "20px"}},
            [html.li(
                {"key": i, "style": todo_item_style},
                html.span(todo),
                Button("Remove", lambda _, i=i: remove_todo(i), "red")
            ) for i, todo in enumerate(todos)]
        ) if todos else html.p("No tasks yet. Add one above!")
    )

# Main App component that composes all other components
@component
def App():
    app_style = {
        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        "maxWidth": "800px",
        "margin": "0 auto",
        "padding": "20px"
    }
    
    return html.div(
        {"style": app_style},
        html.h1(
            {"style": {"color": "#333", "textAlign": "center", "marginBottom": "30px"}},
            "ReactPy Demo Application"
        ),
        html.p(
            {"style": {"textAlign": "center", "marginBottom": "30px", "color": "#666"}},
            "A demonstration of ReactPy components with state management and styling"
        ),
        Card("Interactive Counter", Counter()),
        Card("Todo List Application", TodoList())
    )

# Note: The ReactPy application is now configured in main.py
# We export the App component to be used there
# This allows for proper integration with the FastAPI application