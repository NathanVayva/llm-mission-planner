import pygame
import math
import re
import time
from mission_planner.LLM import OllamaLLM, OpenAILLM
from mission_planner.planner import MissionPlanner
from mission_planner.schemas import MissionPlan

# CONFIGURATION
WIDTH, HEIGHT = 600, 400
FPS = 30
ROBOT_RADIUS = 15
TARGET_RADIUS = 5
ROCK_RADIUS = 10
AnalysisCenter_RADIUS = 20
PHOTOSPOT_RADIUS = 10
ROBOT_MAX_SPEED = 150  # pixels par seconde pour les déplacements LLM

global current_action_index

# CLASSES
class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.angular_speed = 0
        self.target_pos = None
        self.carrying_sample = False  # IMPORTANT
        self.wait_timer = 0  # temps restant à attendre

    

    def step(self, action, dt):

        # si le robot attend, on décrémente le timer et on sort
        global current_action_index
        
        if self.wait_timer > 0:
            self.wait_timer -= dt
            current_action_index-=1
            return

        if action.action == "move_to":
            params = action.parameters or {}
            target_x = float(params.get("target_x", self.x))
            target_y = float(params.get("target_y", self.y))

            # si l'action utilise "waypoint" au lieu de x/y → gérer ça ici
            if "waypoint" in params:
                print("Waypoint donné, mais pas de coordonnées → définir une table de waypoints")
            self.target_pos = (target_x, target_y)

        elif action.action == "pick_up":
            if not self.carrying_sample:
                print("Sample picked up!")
                self.carrying_sample = True

        elif action.action == "drop_off":
            if self.carrying_sample:
                print("Sample delivered to analysis center!")
                self.carrying_sample = False

        elif action.action == "take_photo":
            print("Photo taken!")
            for k, v in (action.parameters or {}).items():
                print(f"  {k} : {v}")
        elif action.action == "wait":
            if self.wait_timer <= 0:
                # initialise le timer à la durée demandée
                self.wait_timer = extract_number((action.parameters or {}).get("duration", 1))
                print(f"Starting wait for {self.wait_timer} seconds...")
            # si le timer est actif, on sort sans rien faire
            return

        # --- déplacement ---
        if self.target_pos:
            dx = self.target_pos[0] - self.x
            dy = self.target_pos[1] - self.y
            dist = math.hypot(dx, dy)

            if dist < 1:
                self.x, self.y = self.target_pos
                self.target_pos = None
            else:
                move_dist = min(ROBOT_MAX_SPEED * dt, dist)
                self.x += (dx / dist) * move_dist
                self.y += (dy / dist) * move_dist
                self.angle = math.atan2(dy, dx)

            

        # ---- Déplacement vers target_pos ----
        if self.target_pos:
            dx = self.target_pos[0] - self.x
            dy = self.target_pos[1] - self.y
            dist = math.hypot(dx, dy)

            if dist < 1:
                self.x, self.y = self.target_pos
                self.target_pos = None
            else:
                move_dist = min(ROBOT_MAX_SPEED * dt, dist)
                self.x += (dx / dist) * move_dist
                self.y += (dy / dist) * move_dist
                self.angle = math.atan2(dy, dx)

    def draw(self, screen):
        color = (0, 200, 255) if not self.carrying_sample else (255, 200, 0)  # change de couleur si échantillon transporté
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), ROBOT_RADIUS)
        end_x = self.x + ROBOT_RADIUS * math.cos(self.angle)
        end_y = self.y + ROBOT_RADIUS * math.sin(self.angle)
        pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 2)


class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), TARGET_RADIUS)  # Analysis Center = vert

class AnalysisCenter:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), (int(self.x), int(self.y)), AnalysisCenter_RADIUS)

class PhotoSpot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 255), (int(self.x), int(self.y)), PHOTOSPOT_RADIUS)

class Rock:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, (150, 150, 150), (int(self.x), int(self.y)), ROCK_RADIUS)


# FONCTIONS UTILES
def apply_llm_action(robot, llm_action, dt):
    robot.step(llm_action, dt)

def extract_number(value, default=1.0):
    if not isinstance(value, str):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", value)
    if match:
        return float(match.group())
    return default

# MAIN
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mini Rover Sample Retrieval")
    clock = pygame.time.Clock()

    global current_action_index
    current_action_index = 0
    


    # --- POSITIONS FIXES ---
    analysis_center = AnalysisCenter(100, 100)
    rock = Rock(400, 250)
    photo_spot = PhotoSpot(500, 350)

    known_locations = {
    "rock": {"x": rock.x, "y": rock.y},
    "analysis_center": {"x": analysis_center.x, "y": analysis_center.y},
    "photo_spot": {"x": photo_spot.x, "y": photo_spot.y}
}

    robot = Robot(analysis_center.x, analysis_center.y)  # le robot démarre ici !



    # --- ACTIONS DU LLM ---
    # Le robot :
    # 1) va au rocher
    # 2) ramasse l'échantillon
    # 3) revient au centre d'analyse
    # 4) dépose l'échantillon
    # llm_actions = [
    #     {"action": "move_to", "parameters": {"target_x": rock.x, "target_y": rock.y}},
    #     {"action": "pick_up_sample", "parameters": {}},
    #     {"action": "move_to", "parameters": {"target_x": analysis_center.x, "target_y": analysis_center.y}},
    #     {"action": "drop_sample", "parameters": {}},
    #     {"action": "move_to", "parameters": {"target_x": photo_spot.x, "target_y": photo_spot.y}}, 
    #     {"action": "take_earth_photo", "parameters": {"resolution": "high", "zoom": "2x"}},
    # ]

    # chat=OllamaLLM(model_name="llama3:instruct")
    # planner = MissionPlanner(chat, model_name="llama3:instruct")

    chat=OllamaLLM(model_name="llama3.2")
    planner = MissionPlanner(chat, model_name="llama3.2")


    mission = """
    Wait for 2 seconds,
    Go to the rock at (rock.x, rock.y),
    pick up a sample,
    return to the analysis center at (analysis_center.x, analysis_center.y),
    drop off the sample,
    then go to the photo spot at (photo_spot.x, photo_spot.y) 
    and take a high-resolution photo with a zoom x10.

    Known coordinates:
    rock = ({} , {})
    analysis_center = ({} , {})
    photo_spot = ({} , {})
    """.format(
    known_locations["rock"]["x"], known_locations["rock"]["y"],
    known_locations["analysis_center"]["x"], known_locations["analysis_center"]["y"],
    known_locations["photo_spot"]["x"], known_locations["photo_spot"]["y"]
    )      


    print("mission:", mission)
    llm_plan = planner.generate_mission_plan(mission)
    llm_actions = llm_plan.actions
    print("LLM Actions:", llm_actions)

    print("Extracted JSON",llm_plan.model_dump_json(indent=2))


    running = True
    while running:
        dt = clock.tick(FPS) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Exécution séquentielle des actions LLM
        if current_action_index < len(llm_actions):
            action = llm_actions[current_action_index]
            apply_llm_action(robot, action, dt)

            # Passe à l'action suivante si:
            # - target_pos == None (donc déplacement terminé)
            # - ou si l'action n'est pas un move
            if robot.target_pos is None:
                current_action_index += 1

        # --- DRAW ---
        screen.fill((220, 220, 220))
        analysis_center.draw(screen)
        robot.draw(screen)
        rock.draw(screen)
        photo_spot.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
