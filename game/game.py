import pygame
import random
import numpy as np
from enum import Enum


class GameState(Enum):
    MENU = 0
    RUNNING = 1
    GAME_OVER = 2


class GameObject:
    """Base class for all game objects"""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)

    def update_rect(self):
        """Update the rectangle position"""
        self.rect.x = self.x
        self.rect.y = self.y

    def draw(self, surface):
        """Draw method to be implemented by subclasses"""
        pass


class Player(GameObject):
    """Player character that runs, jumps and slides"""
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.lane = 1  # 0: left, 1: center, 2: right
        self.is_jumping = False
        self.is_sliding = False
        self.jump_velocity = 0
        self.jump_height = 15
        self.gravity = 1
        self.slide_counter = 0
        self.slide_duration = 30
        self.normal_height = height
        self.slide_height = height // 2
        self.color = (50, 100, 255)  # Blue player
        self.original_y = y

    def move_left(self):
        """Move player to the left lane if possible"""
        if self.lane > 0:
            self.lane -= 1
            return True
        return False

    def move_right(self):
        """Move player to the right lane if possible"""
        if self.lane < 2:
            self.lane += 1
            return True
        return False

    def jump(self):
        """Make the player jump if not already jumping or sliding"""
        if not self.is_jumping and not self.is_sliding:
            self.is_jumping = True
            self.jump_velocity = self.jump_height
            return True
        return False

    def slide(self):
        """Make the player slide if not already jumping or sliding"""
        if not self.is_jumping and not self.is_sliding:
            self.is_sliding = True
            self.slide_counter = 0
            self.height = self.slide_height
            self.y += (self.normal_height - self.slide_height)
            self.update_rect()
            return True
        return False

    def update(self, lane_positions):
        """Update player state based on current actions"""
        # Update x position based on lane
        target_x = lane_positions[self.lane]
        self.x = target_x

        # Handle jumping physics
        if self.is_jumping:
            self.y -= self.jump_velocity
            self.jump_velocity -= self.gravity
            
            # Check if player has landed
            if self.y >= self.original_y:
                self.y = self.original_y
                self.is_jumping = False
                self.jump_velocity = 0

        # Handle sliding timer
        if self.is_sliding:
            self.slide_counter += 1
            if self.slide_counter >= self.slide_duration:
                self.is_sliding = False
                self.height = self.normal_height
                self.y = self.original_y
                self.update_rect()

        # Update rectangle position
        self.update_rect()

    def draw(self, surface):
        """Draw the player character"""
        pygame.draw.rect(surface, self.color, self.rect)


class Obstacle(GameObject):
    """Obstacles the player must avoid"""
    def __init__(self, x, y, width, height, lane, obstacle_type):
        super().__init__(x, y, width, height)
        self.lane = lane
        self.type = obstacle_type  # "barrier", "train", etc.
        self.color = (255, 50, 50) if obstacle_type == "train" else (50, 200, 50)  # Red for trains, green for barriers

    def update(self, speed):
        """Move obstacle downwards"""
        self.y += speed
        self.update_rect()

    def draw(self, surface):
        """Draw the obstacle"""
        pygame.draw.rect(surface, self.color, self.rect)


class Coin(GameObject):
    """Collectible coins"""
    def __init__(self, x, y, lane):
        super().__init__(x, y, 20, 20)  # Coins are 20x20 pixels
        self.lane = lane
        self.collected = False
        self.color = (255, 215, 0)  # Gold color

    def update(self, speed):
        """Move coin downwards"""
        self.y += speed
        self.update_rect()

    def draw(self, surface):
        """Draw the coin"""
        if not self.collected:
            pygame.draw.circle(surface, self.color, (self.x + 10, self.y + 10), 10)


class Powerup(GameObject):
    """Power-ups that give special abilities"""
    def __init__(self, x, y, lane, powerup_type):
        super().__init__(x, y, 30, 30)  # Power-ups are 30x30 pixels
        self.lane = lane
        self.type = powerup_type  # "magnet", "jetpack", etc.
        self.collected = False
        self.color = (150, 50, 200)  # Purple for power-ups

    def update(self, speed):
        """Move power-up downwards"""
        self.y += speed
        self.update_rect()

    def draw(self, surface):
        """Draw the power-up"""
        if not self.collected:
            pygame.draw.rect(surface, self.color, self.rect)
            # Draw a letter indicating powerup type in the center
            font = pygame.font.SysFont(None, 20)
            letter = font.render(self.type[0].upper(), True, (255, 255, 255))
            surface.blit(letter, (self.x + 10, self.y + 8))


class SubwaySurfersEnv:
    """Main game environment"""
    
    # Action space constants
    LEFT = 0
    RIGHT = 1
    JUMP = 2
    SLIDE = 3
    NO_ACTION = 4
    
    def __init__(self, width=600, height=800):
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Display setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Subway Surfers Clone")
        
        # Game state
        self.state = GameState.MENU
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Colors
        self.bg_color = (200, 200, 200)  # Light gray background
        self.text_color = (0, 0, 0)  # Black text
        
        # Game objects
        self.lane_width = width // 3
        self.lane_positions = [
            self.lane_width // 2 - 25,  # Left lane center - half player width
            self.lane_width * 3 // 2 - 25,  # Middle lane center - half player width
            self.lane_width * 5 // 2 - 25  # Right lane center - half player width
        ]
        
        self.player = Player(self.lane_positions[1], height - 150, 50, 80)
        self.obstacles = []
        self.coins = []
        self.powerups = []
        
        # Game parameters
        self.speed = 5
        self.score = 0
        self.distance = 0
        self.high_score = 0
        
        # Spawn timers and rates
        self.obstacle_spawn_rate = 0.02
        self.coin_spawn_rate = 0.05
        self.powerup_spawn_rate = 0.01
        
        # Fonts
        self.font_large = pygame.font.SysFont(None, 48)
        self.font_medium = pygame.font.SysFont(None, 36)
        self.font_small = pygame.font.SysFont(None, 24)
        
        # For RL/GA training
        self.action_space = 5  # LEFT, RIGHT, JUMP, SLIDE, NO_ACTION
        self.observation_space_dim = 30  # Will be calculated based on state representation
        
    def reset(self):
        """Reset the game to initial state"""
        self.player = Player(self.lane_positions[1], self.height - 150, 50, 80)
        self.obstacles = []
        self.coins = []
        self.powerups = []
        self.speed = 5
        self.score = 0
        self.distance = 0
        self.state = GameState.RUNNING
        return self._get_state()
        
    def _spawn_obstacle(self):
        """Randomly spawn obstacles"""
        if random.random() < self.obstacle_spawn_rate:
            lane = random.randint(0, 2)
            obstacle_type = random.choice(["barrier", "train"])
            
            width = 80 if obstacle_type == "train" else 60
            height = 100 if obstacle_type == "train" else 40
            
            x = self.lane_positions[lane]
            y = -height
            
            self.obstacles.append(Obstacle(x, y, width, height, lane, obstacle_type))
    
    def _spawn_coin(self):
        """Randomly spawn coins"""
        if random.random() < self.coin_spawn_rate:
            lane = random.randint(0, 2)
            x = self.lane_positions[lane] + 15  # Center in lane
            y = -20
            self.coins.append(Coin(x, y, lane))
            
    def _spawn_powerup(self):
        """Randomly spawn power-ups"""
        if random.random() < self.powerup_spawn_rate:
            lane = random.randint(0, 2)
            powerup_type = random.choice(["magnet", "jetpack", "multiplier"])
            x = self.lane_positions[lane] + 10  # Center in lane
            y = -30
            self.powerups.append(Powerup(x, y, lane, powerup_type))
            
    def _check_collisions(self):
        """Check for collisions between player and game objects"""
        # Check obstacle collisions
        for obstacle in self.obstacles:
            if self.player.rect.colliderect(obstacle.rect):
                self.state = GameState.GAME_OVER
                return -10  # Negative reward for collision
        
        # Check coin collisions
        reward = 0
        for coin in self.coins:
            if not coin.collected and self.player.rect.colliderect(coin.rect):
                coin.collected = True
                self.score += 1
                reward += 1  # Positive reward for coin
        
        # Check powerup collisions
        for powerup in self.powerups:
            if not powerup.collected and self.player.rect.colliderect(powerup.rect):
                powerup.collected = True
                self.score += 5
                reward += 3  # Positive reward for powerup
                
                # Apply powerup effect (could be expanded)
                if powerup.type == "magnet":
                    pass  # Implement magnet effect
                elif powerup.type == "jetpack":
                    pass  # Implement jetpack effect
                elif powerup.type == "multiplier":
                    pass  # Implement score multiplier
        
        # Small positive reward for surviving
        reward += 0.1
        return reward
        
    def _update_game_state(self):
        """Update the positions and states of all game objects"""
        # Update player
        self.player.update(self.lane_positions)
        
        # Update obstacles
        for obstacle in list(self.obstacles):
            obstacle.update(self.speed)
            # Remove if off-screen
            if obstacle.y > self.height:
                self.obstacles.remove(obstacle)
        
        # Update coins
        for coin in list(self.coins):
            coin.update(self.speed)
            # Remove if off-screen or collected
            if coin.y > self.height or coin.collected:
                self.coins.remove(coin)
        
        # Update powerups
        for powerup in list(self.powerups):
            powerup.update(self.speed)
            # Remove if off-screen or collected
            if powerup.y > self.height or powerup.collected:
                self.powerups.remove(powerup)
        
        # Spawn new objects
        self._spawn_obstacle()
        self._spawn_coin()
        self._spawn_powerup()
        
        # Update game metrics
        self.distance += self.speed / 10
        
        # Increase difficulty over time
        if self.distance % 100 < 0.1:  # Every 100 units of distance
            self.speed = min(self.speed + 0.5, 20)  # Cap at max speed
            self.obstacle_spawn_rate = min(self.obstacle_spawn_rate + 0.005, 0.1)
        
    def _get_state(self):
        """Convert game state to observation for RL/GA training"""
        # Create a grid representation of the game (10 rows x 3 columns)
        state = np.zeros((10, 3), dtype=np.float32)
        
        # Mark player position
        player_row = 9  # Player is always at the bottom
        state[player_row, self.player.lane] = 1
        
        # Mark player state (jumping/sliding)
        if self.player.is_jumping:
            state[player_row, self.player.lane] = 2
        elif self.player.is_sliding:
            state[player_row, self.player.lane] = 3
        
        # Mark obstacles
        for obstacle in self.obstacles:
            # Convert obstacle position to grid position
            row = int((obstacle.y / self.height) * 10)
            if 0 <= row < 10:
                state[row, obstacle.lane] = -1
        
        # Mark coins
        for coin in self.coins:
            if not coin.collected:
                row = int((coin.y / self.height) * 10)
                if 0 <= row < 10:
                    state[row, coin.lane] = 0.5
        
        # Mark powerups
        for powerup in self.powerups:
            if not powerup.collected:
                row = int((powerup.y / self.height) * 10)
                if 0 <= row < 10:
                    state[row, powerup.lane] = 0.8
        
        # Flatten for the neural network input
        return state.flatten()
    
    def step(self, action):
        """Execute action and advance game state"""
        if self.state != GameState.RUNNING:
            return self._get_state(), 0, True, {}
        
        # Process player action
        action_taken = False
        if action == self.LEFT:
            action_taken = self.player.move_left()
        elif action == self.RIGHT:
            action_taken = self.player.move_right()
        elif action == self.JUMP:
            action_taken = self.player.jump()
        elif action == self.SLIDE:
            action_taken = self.player.slide()
        # NO_ACTION does nothing
        
        # Update game state
        self._update_game_state()
        
        # Check for collisions and get reward
        reward = self._check_collisions()
        
        # Additional small negative reward for unused actions to encourage efficiency
        if not action_taken and action != self.NO_ACTION:
            reward -= 0.1
        
        # Get observation
        observation = self._get_state()
        
        # Check if game is over
        done = (self.state == GameState.GAME_OVER)
        
        # Return additional info
        info = {
            "score": self.score,
            "distance": self.distance,
            "speed": self.speed
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the game state"""
        # Fill background
        self.screen.fill(self.bg_color)
        
        # Draw lanes
        for i in range(4):
            x = i * self.lane_width
            pygame.draw.line(self.screen, (100, 100, 100), (x, 0), (x, self.height), 2)
        
        # Draw game objects
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        for coin in self.coins:
            coin.draw(self.screen)
        
        for powerup in self.powerups:
            powerup.draw(self.screen)
        
        self.player.draw(self.screen)
        
        # Draw score and distance
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.text_color)
        distance_text = self.font_medium.render(f"Distance: {int(self.distance)}m", True, self.text_color)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(distance_text, (10, 50))
        
        # Draw game state messages
        if self.state == GameState.MENU:
            title_text = self.font_large.render("Subway Surfers Clone", True, self.text_color)
            start_text = self.font_medium.render("Press SPACE to Start", True, self.text_color)
            self.screen.blit(title_text, (self.width//2 - title_text.get_width()//2, self.height//3))
            self.screen.blit(start_text, (self.width//2 - start_text.get_width()//2, self.height//2))
        
        elif self.state == GameState.GAME_OVER:
            game_over_text = self.font_large.render("Game Over", True, self.text_color)
            restart_text = self.font_medium.render("Press SPACE to Restart", True, self.text_color)
            self.screen.blit(game_over_text, (self.width//2 - game_over_text.get_width()//2, self.height//3))
            self.screen.blit(restart_text, (self.width//2 - restart_text.get_width()//2, self.height//2))
            
            # Show final score
            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.text_color)
            self.screen.blit(final_score_text, (self.width//2 - final_score_text.get_width()//2, self.height//2 + 50))
            
            # Show high score
            if self.score > self.high_score:
                self.high_score = self.score
            high_score_text = self.font_medium.render(f"High Score: {self.high_score}", True, self.text_color)
            self.screen.blit(high_score_text, (self.width//2 - high_score_text.get_width()//2, self.height//2 + 100))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen)
    
    def handle_events(self):
        """Process pygame events for human play"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                # Menu controls
                if self.state == GameState.MENU and event.key == pygame.K_SPACE:
                    self.reset()
                
                # Game over controls
                elif self.state == GameState.GAME_OVER and event.key == pygame.K_SPACE:
                    self.reset()
                
                # In-game controls
                elif self.state == GameState.RUNNING:
                    if event.key == pygame.K_LEFT:
                        self.step(self.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.step(self.RIGHT)
                    elif event.key == pygame.K_UP:
                        self.step(self.JUMP)
                    elif event.key == pygame.K_DOWN:
                        self.step(self.SLIDE)
        
        return True
    
    def close(self):
        """Close the game"""
        pygame.quit()

# Function to play the game manually
def play_game():
    env = SubwaySurfersEnv()
    running = True
    
    while running:
        # Handle events
        running = env.handle_events()
        
        # Update game if running
        if env.state == GameState.RUNNING:
            env.step(env.NO_ACTION)  # Default action is no action
        
        # Render the game
        env.render()
    
    env.close()

if __name__ == "__main__":
    play_game()