from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.db.models import Q

from .models import Duel


def lobby_group(user_id):
    return f"lobby_{user_id}"


def duel_group(duel_id):
    return f"duel_{duel_id}"


class LobbyConsumer(AsyncJsonWebsocketConsumer):
    """One connection per logged-in user on the home page; receives invite events."""

    async def connect(self):
        user = self.scope["user"]
        if not user.is_authenticated:
            await self.close()
            return
        self.group = lobby_group(user.id)
        await self.channel_layer.group_add(self.group, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    async def lobby_event(self, event):
        # Forward the payload to the browser. The event's `data` field is
        # whatever the sender put in it.
        await self.send_json(event["data"])


class DuelConsumer(AsyncJsonWebsocketConsumer):
    """One connection per participant on a duel page; receives progress + end events."""

    async def connect(self):
        user = self.scope["user"]
        if not user.is_authenticated:
            await self.close()
            return
        self.duel_id = self.scope["url_route"]["kwargs"]["duel_id"]
        allowed = await self._is_participant(self.duel_id, user.id)
        if not allowed:
            await self.close()
            return
        self.group = duel_group(self.duel_id)
        await self.channel_layer.group_add(self.group, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    @database_sync_to_async
    def _is_participant(self, duel_id, user_id):
        return Duel.objects.filter(pk=duel_id).filter(
            Q(inviter_id=user_id) | Q(opponent_id=user_id)
        ).exists()

    async def duel_event(self, event):
        await self.send_json(event["data"])
