import pygame
from defaults import *

class TextObject(pygame.sprite.Sprite):
    """Simple class to represent a text object displayed in Menu"""
    def __init__(self, text, x, y, font=None, font_size=25, bold=False, color=GREEN):
        super().__init__()

        self.text = text
        self.x = x
        self.y = y

        # We don't want to open this SysFont too many times
        # it's better to pass it as an argument than create each time
        self.font = font or pygame.font.SysFont('Calibri', font_size, bold, False)

        self.image = self.font.render(text, True, color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def change_text(self, new_text):
        """Changes text in displayed image"""
        self.image = self.font.render(new_text, True, GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = self.x
        self.rect.y = self.y


class Menu:
    """Class responsible for the look of the Menu on the right"""
    def __init__(self, x=SCREEN_WIDTH + 20, y=20):
        # x and y are starting points of menu
        self.x = x
        self.y = y

        # Options
        self.language = EN 
        self.sound = OFF
        self.time = 10
        self.record = "Press 'r' to input a command"                            
        self.move = "Say 'left' or 'right' for direction"
        self.aim = "Say 'up' or 'down' for aiming"
        self.stop = "Say 'stop' for stopping any movement"
        self.shoot = "Say 'yes' for shooting"


        self.font = pygame.font.SysFont('Calibri', 25, False, False)
        self.font_bold = pygame.font.SysFont('Calibri', 25, True, False)
        self.font_time = pygame.font.SysFont('Calibri', 70, True, False)
        self.font_ins = pygame.font.SysFont('Calibri', 11, True, False)

        self.click_sound = pygame.mixer.Sound("sounds/buttons_and_clicks/Clic07.mp3.flac")
        self.change_player_sound = pygame.mixer.Sound("sounds/UI_pack_1/ALERT_Appear.wav")

        self.positions = {
            "menu": (x, y+10),

            "language" : (x, y + 50),
            "language_option" : (x + 100, y + 50),

            "sound" : (x, y + 100),
            "sound_option" : (x + 100, y + 100),
            
            "record" : (x-20, y + 150),
            "move" : (x-20, y + 165),
            "aim" : (x-20, y + 175),
            "stop" : (x-20, y + 185),
            "shoot" : (x-20, y + 195),

            "time" : (x, SCREEN_HEIGHT - 50),
            "time_display" : (x + 100, SCREEN_HEIGHT - 70)
        }

        self.text_objects = {
            "menu": TextObject("MENU", *self.positions["menu"]),

            "language" : TextObject("Language:", *self.positions["language"], self.font),
            "language_option" : TextObject(self.language, *self.positions["language_option"], self.font),

            "sound" : TextObject("Sound:", *self.positions["sound"], self.font),
            "sound_option" : TextObject(self.sound, *self.positions["sound_option"], self.font),
            "record" : TextObject(self.record, *self.positions["record"], self.font_ins),
            "move" : TextObject(self.move, *self.positions["move"], self.font_ins),
            "aim" : TextObject(self.aim, *self.positions["aim"], self.font_ins),
            "stop" : TextObject(self.stop, *self.positions["stop"], self.font_ins),
            "shoot" : TextObject(self.shoot, *self.positions["shoot"], self.font_ins),
            "time" : TextObject("TIME:", *self.positions["time"], font=self.font_bold)
            #"time_display": TextObject(str(0), *self.positions["time_display"], self.font)
        }
        #Tk().wm_withdraw() #to hide the main window
        #messagebox.showinfo(self.instruction,'OK')

    def update_options(self, x, y):
        if self.text_objects["language_option"].rect.collidepoint(x, y):
            if self.sound == ON:
                self.click_sound.play()
            #print("Changing language option")
            new_language_option = EN if self.language == PL else PL
            self.language = new_language_option
            self.text_objects["language_option"] = TextObject(new_language_option, 
                                                              *self.positions["language_option"], 
                                                              self.font)
            if self.language == PL:
                self.text_objects["time"].change_text("CZAS:")
                self.text_objects["language"].change_text("Jezyk:")
                self.text_objects["sound"].change_text("Dzwiek:")

            elif self.language == EN:
                self.text_objects["time"].change_text("TIME:")
                self.text_objects["language"].change_text("Language:")
                self.text_objects["sound"].change_text("Sound:")

        elif self.text_objects["sound_option"].rect.collidepoint(x, y):
            self.click_sound.play()
            #print("Changing sound option")
            new_sound_option = OFF if self.sound == ON else ON
            self.sound = new_sound_option
            self.text_objects["sound_option"] = TextObject(new_sound_option, 
                                                          *self.positions["sound_option"],
                                                          self.font)
            if self.sound == ON:
                pass
                #pygame.mixer.music.play()
            else:
                pygame.mixer.music.pause()

    def update_time(self, seconds, font=None):
        # Make a sound 1 second before it's time to change players
        if self.sound == ON and seconds == 1:
            pass
            #self.change_player_sound.play()
        self.time = seconds
        self.text_objects["time_display"] = TextObject(str(seconds), 
                                                        *self.positions["time_display"], 
                                                        font=self.font_time,
                                                        color=RED)