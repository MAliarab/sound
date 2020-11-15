import cv2
import time


class GridView:
    def __init__(self,rows:int, columns:int, detected_points:list,blink=None,) -> None:

        """
        row: number of horizontal cells
        columns: number of vertical cells
        detected_points: a list of suspicious points
        blink: blink=True/False -> blinking suspicious cells, blink=None -> not blinking
        """
        self.rows = rows
        self.columns = columns
        self.detected_points = detected_points
        self.blink = blink
        
    def show_grid(self,grid_color=(109, 112, 117),rectangle_color=(0,0,255),
                    draw_circle=True,circle_radius=5,circle_color=(0,0,255), alpha=0.4)-> None:
        """
        این تابع جهت نمایش یک گرید بر روی ویدئو لایو می باشد
        با دادن لیستی از نقاط مورد نظر به تابع سلول های شامل نقاط داده شده به حالت چشمک زن در می آید
        پارامترها
        grid_color: color of grid, grid=(B,G,R)
        rectangle_color: color of rectangle, rectangle_color=(B,G,R) 
        draw_circle: boolean for draw a circle around suspecious object
        circle_radius: radius of circle
        circle_color: (B,G,R) - default=(0,0,255), Red
        alpha:float- control transparency degree of suspecious cell, between 0,1
        """
        
        #set video source (default = 0 -> webcam)
        cam = cv2.VideoCapture(0)
        
        while True:
            ret_val, img = cam.read()
            overlay = img.copy()
            img_height, img_width,_ = img.shape
            
            vertical_step = int(img_height/self.rows)
            horizontal_step = int(img_width/self.columns)
            x = horizontal_step
            y = vertical_step

            #draw grid over video
            while x < img_width:
                cv2.line(img,(x,0),(x,img_height),grid_color)
                x += horizontal_step
                
            while y < img_height:
                cv2.line(img,(0,y),(img_width,y),grid_color)
                y += vertical_step 
            # toggle blink if it is not None    
            if self.blink != None:
                    self.blink = not self.blink
                    time.sleep(0.1) # for slower blinking
                
            for i,point in enumerate(self.get_detected_points()):
                
                cell_index = int(point[0]//horizontal_step), int(point[1]//vertical_step)

                if draw_circle:
                    cv2.circle(img,point,circle_radius,circle_color)
                        
                if self.blink == None:
                    cv2.rectangle(overlay, (cell_index[0]*horizontal_step,cell_index[1]*vertical_step),
                                ((cell_index[0]+1)*horizontal_step,(cell_index[1]+1)*vertical_step), rectangle_color, -1)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    
                elif self.blink:
                    cv2.rectangle(overlay, (cell_index[0]*horizontal_step,cell_index[1]*vertical_step),
                                    ((cell_index[0]+1)*horizontal_step,(cell_index[1]+1)*vertical_step), rectangle_color, -1)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            cv2.imshow('grid view (esc to exit)', img)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()

    def set_detected_points(self,new_points:list) -> None :
        self.detected_points = new_points
        
    def get_detected_points(self) -> list:
        return self.detected_points

    def set_blinking(self,blink) -> None:
        self.blink = blink


def main():
    #create a grid with 10 row and 10 columns
    gv = GridView(10,10,[(555,355)],blink=True)
    gv.show_grid()

if __name__ == '__main__':
    main()