"""
This module represents the main entry point of the application 
and the main window of the application

Author: Lim Yun Feng, Ting Yi Xuan, Chua Sheen Wey
Last Edited: 28/10/2023

Components:
    - MainWindow: The main window of the application
    - main: The main (funciton) entry point of the application
"""
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, \
    QLabel, QGridLayout, QBoxLayout, QStackedWidget, QPushButton, \
    QCheckBox, QInputDialog, QVBoxLayout, QFileDialog, QMessageBox

from PyQt5.QtGui import QFont
from PyQt5 import QtCore, QtWidgets
import sys
from datetime import datetime

from utils import get_recording_folder, get_unique_filename, WIDTH
from stream_thread import StreamWorker
from video_source import Drone, Webcam

class MainWindow(QMainWindow):
    """
    This class represents the main window of the application.

    Attributes:
        stack (QStackedWidget): The stacked widget that contains all the pages.
        __stream_type (str): The type of the stream (Webcam or Drone).
        __is_record (bool): Whether the stream should be recorded.
        __is_alert (bool): Whether the stream should show alert.
        __name (str): The name of the recording file.
        __start_stream_button (QPushButton): The start stream button.
        __message_label (QLabel): The message label.
    """

    def __init__(self, stack: QStackedWidget) -> None:
        """
        The constructor for the MainWindow class. Build the UI and oinitialize 
        necessary components of the main window

        Parameters:
            stack (QStackedWidget): The stacked widget that contains all the pages.
        """
        super(MainWindow, self).__init__()
        self.stack = stack
        self.__stream_type = None
        self.__is_record = False
        self.__is_alert = False
        self.__name = None

        # Create an instance of a QBoxLayout layout (main layout).
        main_layout = QVBoxLayout()
        horizontal_magrin = 350
        main_layout.setContentsMargins(horizontal_magrin, 20, horizontal_magrin, 20)
        main_layout.Direction = QBoxLayout.Direction.RightToLeft
        main_layout.sizeConstraint = QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        
        # Sub-function layout
        sub_function_layout = QGridLayout()
        
        # Stream type layout
        type_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        sub_type_layout = QGridLayout()
        
        # Create an instance of a QPushButton class (Sub-function - "Load" and "About").
        load_button = QPushButton("Load Recorded Stream")
        load_button.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        load_button.setFixedHeight(50)
        load_button.clicked.connect(self.open_recordings_folder)

        # Create an instance of a QPushButton class (Stream Type).
        type_stream_label = QLabel('<font size = "4">Streaming Type:</font>')
        
        # Webcam button
        self.webcam_button = QPushButton("Webcam")
        # self.webcam_button.setGeometry(200, 150, 150, 40)
        # self.webcam_button.setStyleSheet("border-radius : 20px; border : 2px solid black")
        self.webcam_button.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.webcam_button.clicked.connect(lambda: self.setStreamType("Webcam"))
        self.webcam_button.setFixedWidth(200)
        self.webcam_button.setFixedHeight(50)

        # Drone button
        self.drone_button = QPushButton("Drone")
        self.drone_button.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.drone_button.clicked.connect(lambda: self.setStreamType("Drone"))
        self.drone_button.setFixedWidth(200)
        self.drone_button.setFixedHeight(50)
        
        # Add every layouts into the main layout
        main_layout.addLayout(sub_function_layout, 0)
        
        # Add widgets into their corresponding layout
        sub_function_layout.addWidget(load_button, 0, 1)

         # Title Label
        title_label = QLabel('<font size = "20"> Real-Time Behaviour Recognition on Surveillance Drone System </font>')
        main_layout.addWidget(title_label, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        title_label.setWordWrap(True)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Add type layout into main layout
        main_layout.addLayout(type_layout, 2)
        type_layout.addWidget(type_stream_label, 0)
        
        # Add sub-type layout into type layout
        type_layout.addLayout(sub_type_layout, 1)

        # Add buttons into sub-type layout 
        sub_type_layout.addWidget(self.webcam_button, 0, 1)
        sub_type_layout.addWidget(self.drone_button, 0, 2)
        
        # Record Checkbox
        record_checkbox = QCheckBox(text="Record the Stream")
        main_layout.addWidget(record_checkbox, 10, QtCore.Qt.AlignmentFlag.AlignHCenter)
        record_checkbox.setFixedSize(400, 50)
        record_checkbox.setFont(QFont("Arial", 10))
        record_checkbox.stateChanged.connect(self.setIsRecord)
        record_checkbox.setContentsMargins(0, 100, 0, 20)
        
        # Alert Checkbox
        self.__alert_checkbox = QCheckBox(text="Show Alert (With Human Tracking)")
        main_layout.addWidget(self.__alert_checkbox, 10, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.__alert_checkbox.setFixedSize(400, 50)
        self.__alert_checkbox.setFont(QFont("Arial", 10))
        self.__alert_checkbox.stateChanged.connect(self.setIsAlert)
        self.__alert_checkbox.setContentsMargins(0, 100, 0, 20)
        

        # Create an instance of a QPushButton class (Start Stream).
        self.__start_stream_button = QPushButton("Start Stream")
        main_layout.addWidget(self.__start_stream_button, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.__start_stream_button.setFixedWidth(400)
        self.__start_stream_button.setFixedHeight(100)
        self.__start_stream_button.setFont(QFont("Arial", 12))
        self.__start_stream_button.clicked.connect(self.startStream)
        self.__start_stream_button.setStyleSheet("background-color : darkgreen; color : white")

        self.__message_label = QLabel('Streaming in progress ...')
        main_layout.addWidget(self.__message_label, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.__message_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.__message_label.setFixedHeight(50)
        self.__message_label.setFont(QFont("Arial", 10))
        self.__message_label.setVisible(False)
        
        # Create a widget instance.
        self.widget = QWidget(self)
        self.widget.setLayout(main_layout)

        # Set the central widget.
        self.setCentralWidget(self.widget)
    
    def startStream(self) -> None:
        """
        Function that is triggered when user clicks on the start stream button.
        """
        # Create a QThreadPool instance.
        pool = QtCore.QThreadPool.globalInstance()
        
        # Identify which stream type
        if self.__stream_type == "Webcam":
            source = Webcam()

        elif self.__stream_type == "Drone":
            source = Drone()

        else:
            # Raise error message and exit if no stream type is selected
            self.__message_label.setVisible(True)
            self.__message_label.setText("Please select a streaming type!")
            return None
        
        # Set up if recording is needed
        self.__name = None
        if self.__is_record:
            # Ask for video name
            self.__name, success = QInputDialog.getText(self, 
                                                 'Request Filename', 
                                                 'Enter a filename for the recording:',
                                                 text=get_unique_filename(self.__stream_type, ".avi")
                                                 )
            if not success:
                return None
            elif self.__name[-4:] != ".avi" and self.__name[-4:] != ".mp4":
                self.__name += ".avi"
                
        # Connect the signals so that the thread can communicate with the GUI.
        camera = StreamWorker(source=source, record=self.__is_record, filename=self.__name, alert=self.__is_alert)
        camera.signals.alert.connect(self.send_alert)
        camera.signals.complete.connect(self.end_stream)

        # Disable the start stream button and show the message label
        self.__start_stream_button.setDisabled(True)
        self.__message_label.setVisible(True)
        self.__message_label.setText("Streaming in progress ...")

        # Start the thread.
        pool.start(camera)
        
    def setStreamType(self, stream_type: str) -> None:
        """
        Function that is triggered when user clicks on the webcam button or drone button.

        Parameters:
            stream_type (str): The type of the stream (Webcam or Drone).
        """
        self.__stream_type = stream_type
        if stream_type == "Webcam":
            self.webcam_button.setStyleSheet("background-color : skyblue")
            self.drone_button.setStyleSheet("background-color : white")
        elif stream_type == "Drone":
            self.drone_button.setStyleSheet("background-color : skyblue")
            self.webcam_button.setStyleSheet("background-color : white")
        
    def setIsRecord(self, record: bool) -> None:
        """	
        Function that is triggered when user clicks on the record checkbox.

        Parameters:
            record (bool): Whether the stream should be recorded.
        """
        self.__is_record = record

    def setIsAlert(self, alert: bool) -> None:
        """
        Function that is triggered when user clicks on the alert checkbox.

        Parameters:
            alert (bool): Whether the stream should show alert.
        """
        self.__is_alert = alert
        
    def open_recordings_folder(self):
        """
        Function that is triggered when user clicks on the load button.

        Launches the file explorer to the recordings folder and open the selected file
        """
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a Recording Video',
            directory=get_recording_folder()
        )
        if response[0] != '':
            os.startfile(response[0])
            
    def send_alert(self, subject_id: int, action: str):
        """
        Function that is triggered when the stream worker thread detects an indecent behaviour.

        Parameters:
            subject_id (int): The subject ID of the person who is performing the indecent behaviour.
            action (str): The indecent behaviour that is performed.
        """   
        # Create a notification window
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)

        # Add the current time to the message
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # Set the message
        msg.setText("Subject ID {} was caught {} at {}".format(subject_id, action, current_time))
        msg.setWindowTitle("Indecent Behaviour Alert")
        
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setGeometry(1200, 1000, 500, 500)
        msg.exec()

    def end_stream(self):
        """
        Function that is triggered when the stream worker thread ends.
        """
        # Enable the start stream button and show the message label
        self.__start_stream_button.setEnabled(True)

        # Show the message label
        if self.__is_record:
            self.__message_label.setText("Filename: '" + self.__name + "' saved successfully!")
        else:
            self.__message_label.setVisible(False)

def main() -> None:
    """	
    The main (function) entry point of the application.
    """
    app = QApplication(sys.argv)

    # Create an instance of a QStackedWidget class.
    stacked_window = QStackedWidget()
    stacked_window.setWindowTitle("MCS23 Surveillance Drone System")
    stacked_window.setFixedWidth(WIDTH)
    stacked_window.setFixedHeight(600)

    # Create an instance of a MainWindow class.
    main_screen = MainWindow(stacked_window)
    stacked_window.addWidget(main_screen)
    
    # Set the current page
    stacked_window.setCurrentIndex(0)
    stacked_window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()