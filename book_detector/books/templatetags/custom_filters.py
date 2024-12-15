import base64
from django import template

register = template.Library()

@register.filter(name='b64encode')
def b64encode(value):
    """Encodes bytes into a base64 string."""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode('utf-8')
    return value
