from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path("ws/lobby/", consumers.LobbyConsumer.as_asgi()),
    path("ws/duel/<int:duel_id>/", consumers.DuelConsumer.as_asgi()),
]
