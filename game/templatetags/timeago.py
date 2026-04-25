from django import template
from django.utils import timezone

register = template.Library()


@register.filter
def timeago(value):
    if value is None:
        return "Prehistoric"

    now = timezone.now()
    delta = now - value
    seconds = int(delta.total_seconds())
    if seconds < 5:
        return "Just now"
    if seconds < 60:
        return f"{seconds} seconds ago"

    minutes = seconds // 60
    if minutes < 60:
        return "1 minute ago" if minutes == 1 else f"{minutes} minutes ago"

    today = timezone.localdate(now)
    that_day = timezone.localdate(value)
    day_diff = (today - that_day).days

    if day_diff == 0:
        hours = seconds // 3600
        return "1 hour ago" if hours == 1 else f"{hours} hours ago"
    if day_diff == 1:
        return "Yesterday"
    if day_diff < 7:
        return f"{day_diff} days ago"
    if day_diff < 14:
        return "Last week"
    if day_diff < 30:
        weeks = day_diff // 7
        return f"{weeks} weeks ago"

    months = (today.year - that_day.year) * 12 + (today.month - that_day.month)
    if months <= 1:
        return "Last month"
    if months < 12:
        return f"{months} months ago"

    years = months // 12
    if years == 1:
        return "Last year"
    return f"{years} years ago"
