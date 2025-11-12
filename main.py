import customtkinter as ctk
from tabs.english_tab import EnglishTab
import lookups

ctk.set_appearance_mode("light")      
ctk.set_default_color_theme("blue")  

class NLPApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(lookups.title)
        self.geometry("1500x900")
        

        self.tabview = ctk.CTkTabview(self, width=1400, height=820)
        self.tabview.pack(pady=20, padx=10, fill="both", expand=True)

        self.tabview.add("English NLP")

        EnglishTab(self.tabview.tab("English NLP"))

if __name__ == "__main__":
    app = NLPApp()
    app.mainloop()
