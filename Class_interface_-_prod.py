import scipy.ndimage.filters
from skimage.restoration import unwrap_phase
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter import font as tkfont
import tkinter.filedialog
import tkinter as tk
import math
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.image as img
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import genfromtxt
import csv
import os.path
import matplotlib
matplotlib.use("TkAgg")


class Interface(tk.Tk):

    def __init__(self, *args, **kwargs):

        self.window = tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Calibri', size=14)

        container = tk.Frame(self, width=1280, height=720)
        container.pack(side="left", fill="both", expand=True)

        self.frames = {}

        list_function = [Main_menu, Reconstruction, Calculate,
                         Average_picture, Phase_difference, Topography,
                         Recovery_distance, Hologram_generation]

        for F in list_function:
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            frame.menu()
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("Main_menu")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class Main_menu(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

    def menu(self):

        tk.Label(self, text="Enter a function",
                 font=self.controller.title_font
                 ).grid(padx=20, pady=15, column=0, row=0)
        tk.Button(self, text="Reconstruction", font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Reconstruction")
                  ).grid(padx=20, pady=10, column=0, row=1)
        tk.Button(self, text="Calculate", font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Calculate")
                  ).grid(padx=20, pady=10, column=0, row=2)
        tk.Button(self, text="Convert CSV to JPG",
                  font=self.controller.title_font,
                  command=convert_csv_to_jpg
                  ).grid(padx=20, pady=10, column=0, row=3)
        tk.Button(self, text="Average picture",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Average_picture")
                  ).grid(padx=20, pady=10, column=0, row=4)
        tk.Button(self, text="Phase difference",
                  font=self.controller.title_font,
                  command=lambda:
                      self.controller.show_frame("Phase_difference")
                  ).grid(padx=20, pady=10, column=0, row=5)
        tk.Button(self, text="Topography", font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Topography")
                  ).grid(padx=20, pady=10, column=0, row=6)
        tk.Button(self, text="Recovery distance",
                  font=self.controller.title_font,
                  command=lambda:
                      self.controller.show_frame("Recovery_distance")
                  ).grid(padx=20, pady=10, column=0, row=7)
        tk.Button(self, text="Hologram generation",
                  font=self.controller.title_font,
                  command=lambda:
                      self.controller.show_frame(
                          "Hologram_generation")
                  ).grid(padx=20, pady=10, column=0, row=8)
        tk.Button(self, text="Exit", font=self.controller.title_font,
                  command=exit_program
                  ).grid(padx=20, pady=30, column=0, row=9)


class Reconstruction(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.holo = np.array([], dtype=float)
        self.name_picture_text_input = tk.StringVar()
        self.wav = 0
        self.pix = 0
        self.dist = 0
        self.N = 0
        self.intensity = np.array([], dtype=float)
        self.intensity_picture = np.array([], dtype=float)
        self.phase = np.array([], dtype=complex)
        self.pix_text_input = tk.StringVar()
        self.size_text_input = tk.StringVar()
        self.wav_text_input = tk.StringVar()
        self.wav_1_text_input = tk.StringVar()
        self.wav_2_text_input = tk.StringVar()
        self.dist_text_input = tk.StringVar()
        self.step_text_input = tk.StringVar()
        self.min_dist_text_input = tk.StringVar()
        self.max_dist_text_input = tk.StringVar()
        self.refwav_1 = np.array([], dtype=float)
        self.refwav_2 = np.array([], dtype=float)
        self.Fresnel = np.array([], dtype=complex)
        self.holo_fresnel = np.array([], dtype=complex)
        self.hologram = np.array([], dtype=float)

    def menu(self):

        tk.Label(self, text="Reconstruction",
                 font=self.controller.title_font
                 ).grid(padx=20, pady=15, column=0, row=0)
        tk.Button(self, text="Open hologram", font=self.controller.title_font,
                  command=self.open_picture_with_title
                  ).grid(padx=20, pady=10, column=0, row=1)
        tk.Button(self, text="Fresnel reconstruction", font=("Calibri", 14),
                  command=self.holography_reconstruction_fresnel
                  ).grid(padx=20, pady=10, column=0, row=5)
        tk.Button(self, text="Convolution reconstruction",
                  font=("Calibri", 14),
                  command=self.holography_reconstruction_convolution
                  ).grid(padx=20, pady=10, column=0, row=6)
        tk.Button(self, text="Back to main menu",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Main_menu")
                  ).grid(padx=20, pady=30, column=0, row=9)

        self.input_data()

    def open_picture_with_title(self):

        if 'name_picture' in globals():
            length_name = len(name_picture[0])
            tk.Label(self, text=(length_name+50)*' ').grid(column=1, row=1)
            open_picture()
            tk.Label(self, text=name_picture).grid(column=1, row=1)
        else:
            open_picture()
            tk.Label(self, text=name_picture).grid(column=1, row=1)

    def input_data(self):

        # Ввод параметров во фрейм (с окном ввода)
        tk.Label(self, text='Wavelength (mm):').grid(column=0, row=2)
        tk.Entry(self, width=20, textvariable=self.wav_text_input
                 ).grid(column=1, row=2)
        tk.Label(self, text='Pixel size (mm):').grid(column=0, row=3)
        tk.Entry(self, width=20, textvariable=self.pix_text_input
                 ).grid(column=1, row=3)
        tk.Label(self, text='Reconstruction distance (mm):'
                 ).grid(column=0, row=4)
        tk.Entry(self, width=20, textvariable=self.dist_text_input).grid(
            column=1, row=4)

    def save_reconstruction(self):

        tk.Label(self, text='Enter intensity and phase filename:').grid(
            padx=20, pady=10, column=0, row=7)
        tk.Entry(self, width=20,
                 textvariable=self.name_picture_text_input
                 ).grid(column=1, row=7)
        tk.Button(self, text="OK", font=("Calibri", 14),
                  command=self.check_image).grid(column=0, row=8)

    def check_image(self):

        # проверка, что файла с такими названиями нет в папке
        if (os.path.exists(self.name_picture_text_input.get()
                           + '_intensity.png')
                or os.path.exists(self.name_picture_text_input.get()
                                  + '_phase.csv')):
            # выводит окно с переименованием
            question = messagebox.askquestion(
                "Warning!", "This name is already in use\nSave file?")
            if question == 'yes':
                self.save_image_and_csv_files()
        else:
            self.save_image_and_csv_files()

    def save_image_and_csv_files(self):
        # сохранение изображения
        if self.name_picture_text_input.get() == '':
            messagebox.showerror('Error!', 'Enter file name!')
        else:
            plt.imsave(self.name_picture_text_input.get() + '_intensity.png',
                       np.power(self.intensity, 0.4), cmap='gray')

            # Сохранение матрицы фазы
            CSV_file = open(self.name_picture_text_input.get() +
                            '_phase.csv', 'w')
            with CSV_file:
                writer = csv.writer(CSV_file, delimiter=';',
                                    lineterminator='\n')
                writer.writerows(self.phase)

            messagebox.showinfo('Reconstruction', 'Reconstruction complete!')

    def demonstrate(self, bright, exp):

        # условие для адаптивной яркости изображения
        if np.mean(self.intensity) < bright:

            br = exp  # показатель степени для яркости

        else:

            br = exp/2

        image_holo_rec = Image.fromarray(np.power(self.intensity, br))
        image_holo_rec = image_holo_rec.resize((400, 400))
        tk.Label(self, text=200*' ').grid(column=2, row=1)
        self.intensity_picture = ImageTk.PhotoImage(image_holo_rec)
        tk.Label(self, image=self.intensity_picture).place(x=650, y=20)

    def calculate_reference_beam(self):

        # Здесь задаётся функция опорного пучка (гаусс)
        self.N = len(self.holo)
        # L = pix * self.N
        gradx_1 = np.zeros((self.N, self.N))
        gradx_sum = 0
        grady_1 = np.zeros((self.N, self.N))
        for j in range(len(self.holo)):
            gradx_sum += 1
            gradx_1[j] = (gradx_sum - self.N / 2) * self.pix
        grady_1 = np.rot90(gradx_1, k=1, axes=(0, 1))
        k = 2 * math.pi / self.wav
        self.refwav_1 = gradx_1 * gradx_1 + grady_1 * grady_1

        return gradx_1, grady_1, self.refwav_1, k

    def reference_beam(self):

        if 'picture' not in globals():
            messagebox.showerror('Error!', 'You forgot to open the picture!')
            return 0
        else:
            if type(picture) == int:
                messagebox.showerror(
                    'Error!', 'You forgot to open the picture!')
                return 0
            else:
                self.holo = -scipy.ndimage.laplace(picture)
                # self.holo = picture

                if (
                        is_number(self.wav_text_input.get()) is not True or
                        is_number(self.pix_text_input.get()) is not True or
                        is_number(self.dist_text_input.get()) is not True or
                        float(self.wav_text_input.get()) == 0 or
                        float(self.pix_text_input.get()) == 0 or
                        float(self.dist_text_input.get()) == 0
                ):
                    messagebox.showerror('Error!', 'Enter correct data!')
                    return 0

                else:

                    self.wav = float(self.wav_text_input.get())
                    self.pix = float(self.pix_text_input.get())
                    self.dist = float(self.dist_text_input.get())

    def function_holography_reconstruction_fresnel(self):

        gradx_1, grady_1, self.refwav_1, k = self.calculate_reference_beam()

        # Преобразование Френеля
        # (преобразование Фурье + смещение спектра к центру)

        # Коэффициент Френеля
        self.Fresnel = np.exp(1j*k/2/self.dist*self.refwav_1)
        self.holo_fresnel = self.holo*self.Fresnel
        Fourier = np.fft.fft2(self.holo_fresnel)  # Преобразование Фурье

        # Смещение нулевой частоты к центру спектра
        Fourier = np.fft.fftshift(Fourier)

        # Размер пикселя при восстановлении
        ipix = self.wav*np.abs(self.dist)/self.N/self.pix
        gradx_2 = gradx_1*ipix/self.pix
        grady_2 = grady_1*ipix/self.pix
        refwav_2 = gradx_2*gradx_2+grady_2*grady_2

        # Коэффициент фазы
        phase_coeff = np.exp(1j*k*self.dist)/(1j*self.wav*self.dist)\
            * np.exp(1j*k/2/self.dist*refwav_2)

        Reconstruction = Fourier*phase_coeff

        self.intensity = np.power(
            np.abs(Reconstruction), 2)  # Интенсивность

        # #применение маски - она домножается на фазу
        # delimiter = define_delimiter('E:/COIL/DHII/Eagle_coin/Eagle/sd_mask.csv')
        # mask = genfromtxt('E:/COIL/DHII/Eagle_coin/Eagle/sd_mask.csv', delimiter=delimiter)

        self.phase = np.arctan2(np.imag(Reconstruction),
                                np.real(Reconstruction))#*mask

        return self.intensity  # это требуется для Recovery_distance

    def holography_reconstruction_fresnel(self):

        if self.reference_beam() == 0:
            pass
        else:
            self.function_holography_reconstruction_fresnel()
            self.demonstrate(500000000, 0.3)
            self.save_reconstruction()

    def function_holography_reconstruction_convolution(self):

        gradx_1, grady_1, refwav_1, k = self.calculate_reference_beam()

        # Метод свёртки (преобразование Фурье + смещение спектра к центру)

        weight = (
            1j*np.exp(
                -1j*k*np.sqrt(np.power(self.dist, 2)+np.power(gradx_1, 2) +
                              np.power(grady_1, 2))
            ) /
            (
                self.wav*np.sqrt(np.power(self.dist, 2)+np.power(gradx_1, 2) +
                                 np.power(grady_1, 2)))
        )

        # Обратное преобразование Фурье весовой функции
        weight_f = np.fft.ifft2(weight)
        # Смещение нулевой частоты весовой функции к центру спектра
        weight_f = np.fft.ifftshift(weight_f)

        # Обратное преобразование Фурье голограммы
        holo_f = np.fft.ifft2(self.holo)
        # Смещение нулевой частоты голограммы к центру спектра
        holo_f = np.fft.ifftshift(holo_f)

        # свёртка весовой функции и голограммы
        conv = np.fft.fft2(weight_f*holo_f)
        # смещение нулевой частоты свёртки к центру спектра
        conv = np.fft.fftshift(conv)

        # Коэффициент фазы
        phase_coeff = np.exp(1j*k*self.dist)/(1j*self.wav*self.dist)\
            * np.exp(1j*k/2/self.dist*refwav_1)

        Reconstruction = conv*phase_coeff

        self.intensity = np.power(
            np.abs(Reconstruction), 2)  # Интенсивность

        self.phase = np.arctan2(np.imag(Reconstruction),
                                np.real(Reconstruction))

    def holography_reconstruction_convolution(self):

        if self.reference_beam() == 0:
            pass
        else:
            self.function_holography_reconstruction_convolution()
            self.demonstrate(2000, 1)
            self.save_reconstruction()


class Calculate(Reconstruction):

    def menu(self):

        tk.Label(self, text="Calculate synthethic wavelength and object size",
                 font=self.controller.title_font
                 ).grid(padx=20, pady=15, column=0, row=0)
        tk.Button(self, text="Calculate", font=("Calibri", 14),
                  command=self.print_data
                  ).grid(padx=20, pady=10, column=0, row=5)
        tk.Button(self, text="Back to main menu",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Main_menu")
                  ).grid(padx=20, pady=30, column=0, row=9)
        self.input_data()

    def input_data(self):

        # Ввод параметров во фрейм (с окном ввода)
        tk.Label(self, text='Wavelength 1 (mm):').grid(column=0, row=1)
        tk.Entry(self, width=20, textvariable=self.wav_1_text_input
                 ).grid(column=1, row=1)
        tk.Label(self, text='Wavelength 2 (mm):').grid(column=0, row=2)
        tk.Entry(self, width=20, textvariable=self.wav_2_text_input
                 ).grid(column=1, row=2)
        tk.Label(self, text='Pixel size (mm):').grid(column=0, row=3)
        tk.Entry(self, width=20, textvariable=self.pix_text_input
                 ).grid(column=1, row=3)
        tk.Label(self, text='Reconstruction distance (mm):'
                 ).grid(column=0, row=4)
        tk.Entry(self, width=20, textvariable=self.dist_text_input
                 ).grid(column=1, row=4)

    def calculate_data(self):

        if (
            is_number(self.wav_1_text_input.get()) is not True or
            is_number(self.wav_2_text_input.get()) is not True or
            is_number(self.pix_text_input.get()) is not True or
            is_number(self.dist_text_input.get()) is not True or
            float(self.wav_1_text_input.get()) == 0 or
            float(self.wav_2_text_input.get()) == 0 or
            float(self.pix_text_input.get()) == 0 or
            float(self.dist_text_input.get()) == 0 or
            float(self.wav_1_text_input.get()) == float(
                self.wav_2_text_input.get())
        ):
            messagebox.showerror('Error!', 'Enter correct data!')
            return 0, 0

        else:

            wav_1 = float(self.wav_1_text_input.get())
            wav_2 = float(self.wav_2_text_input.get())
            self.pix = float(self.pix_text_input.get())
            self.dist = float(self.dist_text_input.get())

            width = np.abs(wav_1*self.dist/self.pix)
            # формула нахождения длины волны биений
            synt_wav = np.abs(wav_1*wav_2/(wav_1-wav_2))

            return width, synt_wav

    def print_data(self):

        width, synt_wav = self.calculate_data()
        if width == 0 and synt_wav == 0:
            pass

        else:
            # считывает обработанные переменные
            #   и преобразует их в текстовые данные с округлением
            width_text = str(round(width, 3))
            synt_wav_text = str(round(synt_wav, 4))
            rough_depth_text = str(round(synt_wav/2, 4))

            width_string = 'Image width = ' + width_text + 'mm'
            synt_wav_string = 'Syntethic wave = ' + synt_wav_text + 'mm'
            rough_depth_string = 'Roughness depth = ' +\
                rough_depth_text + 'mm'

            tk.Label(self, text=synt_wav_string).grid(column=0, row=6)
            tk.Label(self, text=width_string).grid(column=0, row=7)
            tk.Label(self, text=rough_depth_string).grid(column=0, row=8)


class Average_picture(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

    def menu(self):

        tk.Label(self, text="Average picture",
                 font=self.controller.title_font
                 ).grid(padx=20, pady=15, column=0, row=0)
        tk.Button(self, text="CSV files", font=("Calibri", 14),
                  command=self.csv_average_picture
                  ).grid(padx=20, pady=10, column=0, row=1)
        tk.Button(self, text="Image files", font=("Calibri", 14),
                  command=self.image_average_picture
                  ).grid(padx=20, pady=10, column=0, row=2)
        tk.Button(self, text="Back to main menu",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Main_menu")
                  ).grid(padx=20, pady=30, column=0, row=3)

    def csv_average_picture(self):

        # выбор директории
        directory = tk.filedialog.askdirectory()
        if directory == '':
            messagebox.showerror('Error!', 'Format file error!')
        else:
            # массив с матрицами в формате *.csv
            all_picture = [f for f in os.listdir(directory) if f.endswith(
                '.csv')]
            count = 0
            # если файлов нет, вывести ошибку
            if all_picture == []:
                messagebox.showerror('Error!', 'No such files!')
            else:
                delimiter = define_delimiter(directory+'/'+all_picture[0])
                # число пикселей голограммы по горизонтали
                N = len(genfromtxt(directory+'/' +
                        all_picture[0], delimiter=delimiter))
                for i in range(len(all_picture)):
                    # определение числа пикселей
                    #   отдельного изображения в массиве изображений
                    N_local = len(genfromtxt(
                        directory+'/'+all_picture[i], delimiter=delimiter))
                    # если есть изображения, которые отличаются по размеру,
                    #   то выдать ошибку
                    if N_local != N:
                        messagebox.showerror(
                            'Error!', 'Images have different resolutions')
                        return 0

                sum = np.zeros((N, N))
                for i in range(len(all_picture)):
                    picture = genfromtxt(
                        directory+'/'+all_picture[i], delimiter=delimiter)
                    N = len(picture)
                    # обрезка изображения до квадратного вида
                    picture = picture[:, :N]
                    sum += picture
                    count += 1
                # среднее арифметическое изображений
                average_picture = sum / count
                plt.imsave(directory+'/'+'average.png',
                           average_picture, cmap='gray')
                CSV_file = open(directory+'/'+'average.csv', 'w')
                with CSV_file:
                    writer = csv.writer(CSV_file, delimiter=';',
                                        lineterminator='\n')
                    writer.writerows(average_picture)
                # вывод информационного сообщения о завершении усреднения
                messagebox.showinfo('Averaging', 'Averaging complete!')

    # подфункция усреднения нескольких голограмм в формате изображений

    def image_average_picture(self):
        # выбор директории
        directory = tk.filedialog.askdirectory()
        if directory == '':
            messagebox.showerror('Error!', 'Format file error!')
        else:
            # массив с матрицами в формате png, jpg или bmp
            all_picture = [f for f in os.listdir(directory)
                           if f.endswith('.png') or
                           f.endswith('.jpg') or f.endswith('.bmp')]
            count = 0
            if all_picture == []:
                messagebox.showerror('Error!', 'No such files!')
            else:
                # число пикселей голограммы по горизонтали
                N = len(img.imread(directory+'/'+all_picture[0]))
                for i in range(len(all_picture)):
                    # определение числа пикселей отдельного
                    #   изображения в массиве изображений
                    N_local = len(img.imread(directory+'/'+all_picture[i]))
                    # если есть изображения, которые отличаются по размеру,
                    #   то выдать ошибку
                    if N_local != N:
                        messagebox.showerror(
                            'Error!', 'Images have different resolutions')
                        return 0

                sum = np.zeros((N, N))
                picture = np.zeros((N, N))
                picture_1 = np.zeros((N, N))
                for i in range(len(all_picture)):
                    picture = img.imread(directory+'/'+all_picture[i])
                    N = len(picture)
                    # обрезка изображений до квадрата и преобразование
                    #   значений интенсивности в тип float
                    np.array(picture[:, :N], dtype=float)
                    if picture.ndim == 3:
                        # выделение одного цветового канала,
                        #   если изображение имеет три цветовых канала
                        picture_1 = picture[:, :, 0]
                    else:
                        # если цветовой канал 1, то изображение не меняется
                        picture_1 = picture
                    sum += picture_1
                    count += 1

                # среднее арифметическое изображений
                average_picture = sum / count
                plt.imsave(directory+'/'+'average.png',
                           average_picture, cmap='gray')
                CSV_file = open(directory+'/'+'average.csv', 'w')
                with CSV_file:
                    writer = csv.writer(CSV_file, delimiter=';',
                                        lineterminator='\n')
                    writer.writerows(average_picture)
                # вывод информационного сообщения о завершении усреднения
                messagebox.showinfo('Averaging', 'Averaging complete!')


class Phase_difference(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.name_phase_diff_text_input = tk.StringVar()

    def menu(self):

        tk.Label(self, text="Phase difference",
                 font=self.controller.title_font
                 ).grid(padx=20, pady=15, column=0, row=0)
        tk.Label(self, text='Enter phase difference filename:'
                 ).grid(padx=20, pady=10, column=0, row=1)
        tk.Entry(self, width=20, textvariable=self.name_phase_diff_text_input
                 ).grid(padx=20, pady=10, column=1, row=1)
        tk.Button(self, text="Without filtering", font=("Calibri", 14),
                  command=self.phase_difference_without_filtering
                  ).grid(padx=20, pady=10, column=0, row=4)
        tk.Button(self, text="With filtering", font=("Calibri", 14),
                  command=self.phase_difference_with_filtering
                  ).grid(padx=20, pady=10, column=0, row=5)
        tk.Button(self, text="Back to main menu",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Main_menu")
                  ).grid(padx=20, pady=30, column=0, row=6)

    def open_phase(self):

        phase_1 = open_picture()[0]
        tk.Label(self, text=name_picture).grid(column=1, row=2)
        phase_2 = open_picture()[0]
        tk.Label(self, text=name_picture).grid(column=1, row=3)
        return phase_1, phase_2

    def phase_subtraction(self):

        if 'name_picture' in globals():
            length_name = len(name_picture[0])
            tk.Label(self, text=(length_name+50)*' ').grid(column=1, row=2)
            tk.Label(self, text=(length_name+50)*' ').grid(column=1, row=3)
            phase_1, phase_2 = self.open_phase()
        else:
            phase_1, phase_2 = self.open_phase()

        if type(phase_1) == int or type(phase_2) == int:
            messagebox.showerror('Error!', 'Open image or CSV files!')
            return 0
        else:
            if np.shape(phase_1) != np.shape(phase_2):
                messagebox.showerror(
                    'Error!', 'Images have different resolutions')
                return 0
            else:
                phase_diff = np.zeros((len(phase_1), len(phase_1)))
                # разность фаз
                for i in range(len(phase_1)):
                    for j in range(len(phase_1[i])):
                        if phase_2[i][j]-phase_1[i][j] < 0:
                            phase_diff[i][j] = phase_2[i][j] - \
                                phase_1[i][j] + 2*math.pi
                        else:
                            phase_diff[i][j] = phase_2[i][j]-phase_1[i][j]

                return phase_diff

    def phase_difference_without_filtering(self):

        phase_diff = self.phase_subtraction()

        if type(phase_diff) == int:
            pass
        else:
            if self.name_phase_diff_text_input.get() == '':
                messagebox.showerror('Error!', 'Enter file name!')
            else:
                plt.imsave(self.name_phase_diff_text_input.get() +
                           '.png', phase_diff, cmap='gray')
                CSV_file = open(
                    self.name_phase_diff_text_input.get() + '.csv', 'w')
                with CSV_file:
                    writer = csv.writer(CSV_file, delimiter=';',
                                        lineterminator='\n')
                    writer.writerows(phase_diff)

                messagebox.showinfo('Save phase difference',
                                    'Phase difference complete!')

    def phase_difference_with_filtering(self):

        phase_diff = self.phase_subtraction()

        phase_diff_filtered = scipy.ndimage.median_filter(
            phase_diff, size=(1, 1))

        for k in range(50):  # алгоритм sin-cos фильтрации
            sin_phase_diff = np.sin(phase_diff_filtered)
            cos_phase_diff = np.cos(phase_diff_filtered)
            sin_phase_diff_filtered = scipy.ndimage.median_filter(
                sin_phase_diff, size=(5, 5))
            cos_phase_diff_filtered = scipy.ndimage.median_filter(
                cos_phase_diff, size=(5, 5))
            phase_diff_filtered = np.arctan2(
                cos_phase_diff_filtered, sin_phase_diff_filtered)

        plt.imsave(self.name_phase_diff_text_input.get() +
                   '_filtered.png', phase_diff_filtered, cmap='gray')

        CSV_file = open(
            self.name_phase_diff_text_input.get() + '_filtered.csv', 'w')
        with CSV_file:
            writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
            writer.writerows(phase_diff_filtered)

        phase_diff_unwrapped = unwrap_phase(
            phase_diff_filtered)  # функция развертки фазы

        plt.imsave(self.name_phase_diff_text_input.get() +
                   '_unwrapped_filtered.png',
                   phase_diff_unwrapped, cmap='gray')

        CSV_file = open(self.name_phase_diff_text_input.get() +
                        '_unwrapped_filtered.csv', 'w')
        with CSV_file:
            writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
            writer.writerows(phase_diff_unwrapped)

        messagebox.showinfo('Save phase difference',
                            'Phase difference complete!')


class Topography(Calculate):

    def menu(self):

        tk.Label(self, text="Topography",
                 font=self.controller.title_font
                 ).grid(padx=20, pady=15, column=0, row=0)
        Calculate.input_data(self)
        tk.Button(self, text="Window interface", font=("Calibri", 14),
                  command=self.window_topography
                  ).grid(padx=20, pady=10, column=0, row=5)
        tk.Button(self, text="TKinter interface", font=("Calibri", 14),
                  command=self.tkinter_topography
                  ).grid(padx=20, pady=10, column=1, row=5)
        tk.Button(self, text="Back to main menu",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Main_menu")
                  ).grid(padx=20, pady=30, column=0, row=6)

    def topography_calculate(self, width, synt_wav):

        depth_map = picture*synt_wav/(4*math.pi)

        N = len(depth_map)
        # задание линейной функции возрастания для двух других осей объекта
        x, y = np.meshgrid(np.arange(N), np.arange(N))

        return depth_map, width, synt_wav, N, x, y

    def window_topography(self):

        width, synt_wav = Calculate.calculate_data(self)
        if width == 0 or synt_wav == 0:
            pass
        else:
            if open_picture() == 0:
                pass
            else:
                depth_map, width, synt_wav, N, x, y = \
                    self.topography_calculate(width, synt_wav)

                plt.rcParams['font.size'] = '12'

                axes = plt.subplot(111, projection='3d')

                # подписи осей графика
                axes.set_xlabel('Width,mm', fontsize=12)
                axes.set_ylabel('Width,mm', fontsize=12)
                axes.set_zlabel('Depth,mm', fontsize=12)

                # трёхмерная визуализация поверхности
                axes.plot_surface(x*width/N, y*width/N, depth_map,
                                  cmap=cm.coolwarm,
                                  linewidth=0, antialiased=False)

                plt.show()

    def tkinter_topography(self):

        width, synt_wav = Calculate.calculate_data(self)
        if width == 0 or synt_wav == 0:
            pass
        else:
            if open_picture() == 0:
                pass
            else:
                depth_map, width, synt_wav, N, x, y = \
                    self.topography_calculate(width, synt_wav)

                # параметры окна для графика
                fig = Figure(figsize=(5, 4), dpi=100)
                # задаёт холст для графика
                canvas = FigureCanvasTkAgg(fig, self)

                # отрисовка холста
                canvas.draw()

                # положение холста
                tk.Label(self, text=200*' ').grid(column=2, row=0)
                canvas.get_tk_widget().place(x=400, y=20)

                # задаёт оси графика
                axes = fig.add_subplot(111, projection='3d')

                # подписи осей графика
                axes.set_xlabel('Width,mm', fontsize=12)
                axes.set_ylabel('Width,mm', fontsize=12)
                axes.set_zlabel('Depth,mm', fontsize=12)

                # трёхмерная визуализация поверхности
                axes.plot_surface(x*width/N, y*width/N, depth_map,
                                  cmap=cm.coolwarm,
                                  linewidth=0, antialiased=False)


class Recovery_distance(Reconstruction):

    def menu(self):

        tk.Label(self, text="Recovery distance",
                 font=self.controller.title_font
                 ).grid(padx=20, pady=15, column=0, row=0)
        tk.Button(self, text="Open hologram", font=("Calibri", 14),
                  command=self.open_picture_with_title
                  ).grid(padx=20, pady=10, column=0, row=1)
        self.input_data()
        tk.Label(self, text='Min reconstruction distance (mm):'
                 ).grid(padx=20, column=0, row=4)
        tk.Entry(self, width=20, textvariable=self.dist_text_input
                 ).grid(padx=20, column=1, row=4)
        tk.Label(self, text='Max reconstruction distance (mm):'
                 ).grid(padx=20, column=0, row=5)
        tk.Entry(self, width=20, textvariable=self.max_dist_text_input
                 ).grid(padx=20, column=1, row=5)
        tk.Label(self, text='Step distance reconstruction  (mm):'
                 ).grid(padx=20, column=0, row=6)
        tk.Entry(self, width=20, textvariable=self.step_text_input
                 ).grid(padx=20, column=1, row=6)
        # запуск функции голографического восстановления методом Френеля
        tk.Button(self, text="Enter", font=("Calibri", 14),
                  command=self.recovery_distance_definition
                  ).grid(padx=20, pady=10, column=1, row=7)
        tk.Button(self, text="Back to main menu",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Main_menu")
                  ).grid(padx=20, pady=30, column=0, row=10)

    def recovery_distance_definition(self):

        if self.reference_beam() == 0:
            pass
        else:

            if (
                    is_number(self.max_dist_text_input.get()) is not True or
                    is_number(self.step_text_input.get()) is not True or
                    float(self.max_dist_text_input.get()) == 0 or
                    float(self.step_text_input.get()) == 0 or
                    float(self.dist_text_input.get()) > float(
                        self.max_dist_text_input.get()) or
                    float(self.step_text_input.get()) > float(
                        self.max_dist_text_input.get())
            ):
                messagebox.showerror('Error!', 'Enter correct data!')

            else:

                # начальная дистанция восстановления
                min_dist = self.dist
                # конечная дистанция восстановления
                max_dist = float(self.max_dist_text_input.get())
                step = float(self.step_text_input.get())

                # нулевой массив с числом элементов,
                #   равным количеству шагов по дистанции
                array_intensity = np.zeros(
                    (round((max_dist - min_dist) / step) + 1))

                # выводит окно с переименованием
                question = messagebox.askquestion(
                    "Attention", "Save reconstruction pictures?")

                # цикл восстановления с разными шагами
                for i in range(len(array_intensity)):

                    self.intensity = \
                        self.function_holography_reconstruction_fresnel()

                    # Сохранение изображений

                    if question == 'yes':
                        plt.imsave(str(self.dist) + '_intensity.png',
                                   np.power(self.intensity, 0.4), cmap='gray')

                    # Определение средней интенсивности

                    self.intensity = self.intensity / np.max(self.intensity)

                    # подсчёт суммы интенсивностей всех пикселей
                    #   восстановленного изображения
                    sum_intensity = np.sum(np.sum(self.intensity))
                    # заполнение массива с суммами
                    array_intensity[i] = sum_intensity

                    # следующий шаг
                    self.dist = self.dist + step

                # Определение дистанции восстановления

                # преобразование numpy.array в list
                array_intensity = array_intensity.tolist()

                # индекс минимального элемента массива
                index_min_intensity = array_intensity.index(
                    min(array_intensity))

                dist_corr_reconsctruction = min_dist+index_min_intensity*step
                dist_corr_reconsctruction_text = str(dist_corr_reconsctruction)
                dist_corr_reconsctruction_str = 'Recovery distance (mm): ' + \
                    dist_corr_reconsctruction_text + 'mm'
                tk.Label(self, text=dist_corr_reconsctruction_str
                         ).grid(padx=20, column=0, row=8)

                # теперь минимальному элементу массива присваивается значение,
                #   например, максимального элемента,
                #       чтобы найти второй с конца минимальный элемент
                array_intensity[index_min_intensity] = max(array_intensity)
                index_min_intensity = array_intensity.index(
                    min(array_intensity))

                dist_corr_reconsctruction = min_dist+index_min_intensity*step
                dist_corr_reconsctruction_text = str(dist_corr_reconsctruction)
                dist_corr_reconsctruction_str = \
                    'Extra distance recovery (mm): ' \
                    + dist_corr_reconsctruction_text + 'mm'
                tk.Label(self, text=dist_corr_reconsctruction_str) \
                    .grid(padx=20, column=0, row=9)


class Hologram_generation(Reconstruction):

    def menu(self):

        tk.Label(self, text='Hologram generation', font=("Calibri", 14)
                 ).grid(padx=20, pady=15, column=0, row=0)
        tk.Button(self, text="Open picture", font=("Calibri", 14),
                  command=self.open_picture_with_title
                  ).grid(padx=20, pady=10, column=0, row=1)
        tk.Button(self, text="Back to main menu",
                  font=self.controller.title_font,
                  command=lambda: self.controller.show_frame("Main_menu")
                  ).grid(padx=20, pady=30, column=0, row=8)
        tk.Label(self, text='Wavelength (mm):').grid(padx=20, column=0, row=2)
        tk.Entry(self, width=20, textvariable=self.wav_text_input
                 ).grid(column=1, row=2)
        tk.Label(self, text='Object size (mm):').grid(padx=20, column=0, row=3)
        tk.Entry(self, width=20, textvariable=self.size_text_input
                 ).grid(column=1, row=3)
        tk.Label(self, text='Reconstruction distance (mm):'
                 ).grid(padx=20, column=0, row=4)
        tk.Entry(self, width=20, textvariable=self.dist_text_input
                 ).grid(column=1, row=4)
        tk.Button(self, text="Enter", font=("Calibri", 14),
                  command=self.function_hologram_generation
                  ).grid(padx=20, pady=10, column=1, row=5)

    def function_hologram_generation(self):

        if (
                is_number(self.wav_text_input.get()) is not True or
                is_number(self.dist_text_input.get()) is not True or
                is_number(self.size_text_input.get()) is not True or
                float(self.wav_text_input.get()) == 0 or
                float(self.dist_text_input.get()) == 0 or
                float(self.size_text_input.get()) == 0
        ):
            messagebox.showerror('Error!', 'Enter correct data!')

        else:

            self.wav = float(self.wav_text_input.get())
            self.size = float(self.size_text_input.get())
            self.dist = float(self.dist_text_input.get())

            N = len(picture)
            self.pix = np.abs(self.wav*self.dist/self.size)

            gradx_1 = np.zeros((N, N))
            gradx_sum = 0
            grady_1 = np.zeros((N, N))
            for j in range(len(picture)):
                gradx_sum += 1
                gradx_1[j] = (gradx_sum - N / 2) * self.pix
            grady_1 = np.rot90(gradx_1, k=1, axes=(0, 1))
            k = 2 * math.pi / self.wav
            refwav_1 = gradx_1 * gradx_1 + grady_1 * grady_1

            ipix = self.wav*np.abs(self.dist)/N/self.pix
            gradx_2 = gradx_1*ipix/self.pix
            grady_2 = grady_1*ipix/self.pix
            refwav_2 = gradx_2*gradx_2+grady_2*grady_2

            # Коэффициент Френеля
            self.Fresnel = np.exp(1j*k/2/self.dist*refwav_1)
            self.holo_fresnel = picture*self.Fresnel
            Fourier = np.fft.fft2(self.holo_fresnel)  # Преобразование Фурье

            # Смещение нулевой частоты к центру спектра
            Fourier = np.fft.fftshift(Fourier)

            phase_coeff = np.exp(1j*k*self.dist)/(1j*self.wav*self.dist)\
                * np.exp(1j*k/2/self.dist*refwav_2)

            self.hologram_before = Fourier*phase_coeff

            # self.coef = np.exp(1j*k*self.dist)/(1j*self.wav*self.dist)\
            #     * np.exp(1j*k/2/self.dist*refwav_2)

            # self.return_fourier = picture * \
            #     np.exp(1j*k/2/self.dist*refwav_2)

            # self.hologram_before = self.coef * \
            #     np.fft.fft2(self.return_fourier)  # Преобразование Фурье

            # # Смещение нулевой частоты к центру спектра
            # self.hologram_before = np.fft.fftshift(self.hologram_before)

            self.hologram = np.power(
                np.abs(self.hologram_before+refwav_1*np.exp(-1j*2*math.pi*0.05/self.wav)), 2)  # Интенсивность

            # self.N = len(picture)
            # size_matrix = 2 * self.N
            # zeros_1 = np.zeros((size_matrix, (size_matrix - self.N) // 2))
            # zeros_2 = np.zeros(((size_matrix - self.N) // 2, self.N))

            # object_holo_aver = np.vstack(
            #     (np.vstack((zeros_2, picture)), zeros_2))
            # object_holo = np.hstack(
            #     (np.hstack((zeros_1, object_holo_aver)), zeros_1))

            # pix = np.abs(self.dist)*self.wav/self.size
            # size_x = pix * size_matrix

            # phase_rand = 2*math.pi \
            #     * (np.random.rand(size_matrix, size_matrix)-0.5)
            # complex_obj = object_holo*np.exp(1j*phase_rand)

            # gradx_1 = np.zeros((size_matrix, size_matrix))
            # gradx_sum = 0
            # grady_1 = np.zeros((size_matrix, size_matrix))
            # for j in range(len(picture)):
            #     gradx_sum += 1
            #     gradx_1[j] = (gradx_sum-size_matrix/2)*pix
            # grady_1 = np.rot90(gradx_1, k=1, axes=(0, 1))
            # k = 2 * math.pi/self.wav
            # refwav_1 = gradx_1*gradx_1+grady_1*grady_1

            # Fresnel = np.exp(1j*k/2/self.dist*refwav_1)
            # obj_fresnel = complex_obj*Fresnel

            # # Преобразование Фурье
            # Fourier = np.fft.fft2(obj_fresnel)
            # # Смещение нулевой частоты к центру спектра
            # Fourier = np.fft.fftshift(Fourier)

            # # Размер пикселя при восстановлении
            # ipix = self.wav * np.abs(self.dist) / size_matrix / pix
            # gradx_2 = gradx_1 * ipix / pix
            # grady_2 = grady_1 * ipix / pix
            # refwav_2 = gradx_2 * gradx_2 + grady_2 * grady_2

            # # Коэффициент фазы
            # phase_coeff = np.exp(1j*k*self.dist)/(1j*self.wav*self.dist) \
            #     * np.exp(1j*k/2/self.dist*refwav_2)
            # hologram_before = Fourier*phase_coeff
            # freq = size_x/8/self.wav/self.dist
            # amplitude = np.max(np.max(np.abs(hologram_before)))

            # ref_wave = amplitude \
            #     * np.exp(2*1j*math.pi*freq*(gradx_1+gradx_2))
            # hologram_med = np.power(np.abs(ref_wave+hologram_before), 2)
            # max_hologram_obj = np.max(np.max(hologram_med))

            # self.hologram = 255*hologram_med/max_hologram_obj

            tk.Label(self, text='Enter hologram filename:'
                     ).grid(column=0, row=6)
            tk.Entry(self, width=20, textvariable=self.name_picture_text_input
                     ).grid(column=1, row=6)

            # вызов функции сохранения восстановленных голограмм
            tk.Button(self, text="OK", font=("Calibri", 14),
                      command=self.save_image_and_csv_hologram
                      ).grid(column=1, row=7)

    def save_image_and_csv_hologram(self):

        if self.name_picture_text_input.get() == '':
            messagebox.showerror('Error!', 'Enter file name!')
        else:
            plt.imsave('hologram_'+self.name_picture_text_input.get()+'.png',
                       self.hologram[self.N:, self.N:], cmap='gray')

            # Сохранение матрицы фазы
            CSV_file = open(
                'hologram_'+self.name_picture_text_input.get()+'.csv', 'w')
            with CSV_file:
                writer = csv.writer(CSV_file, delimiter=';',
                                    lineterminator='\n')
                writer.writerows(self.hologram[self.N:, self.N:])

            # сообщение на экране, что реконструкция прошла успешно
            messagebox.showinfo('Hologram generation',
                                'Hologram generation complete!')


file_extension = ['.jpg', '.png', '.bmp']


def open_picture():

    global picture
    global name_picture
    global name_picture_open

    # вызываем диалог открытия файла
    name_picture_open = tk.filedialog.askopenfilename()
    name_picture = os.path.splitext(name_picture_open)

    # делим функцией os.path.splitext() название файла на имя и расширение,
    #   и сравниваем расширение с указанным
    if os.path.splitext(name_picture_open)[1] == '.csv':
        # преобразуем csv в двумерный массив
        delimiter = define_delimiter(name_picture_open)
        picture_open = genfromtxt(name_picture_open, delimiter=delimiter)
        # высота картинки в пикселях
        N = len(picture_open)
        # обрезаем ширину картинки до высоты
        #   (если высота больше ширины, то произойдет ошибка)
        picture = picture_open[:, :N]

    elif os.path.splitext(name_picture_open)[1].lower() in file_extension:

        # преобразуем картинку в массив
        picture_open = img.imread(name_picture_open)
        if np.shape(picture_open)[0] > np.shape(picture_open)[1]:
            picture_open = np.rot90(picture_open, k=1, axes=(0, 1))
        N = len(picture_open)
        picture_open = np.array(picture_open[:, :N], dtype=float)

        # если размерность массива 3, то используем один канал
        if picture_open.ndim == 3:
            picture = 0.299*picture_open[:, :, 0] + 0.587 \
                * picture_open[:, :, 1] + 0.114*picture_open[:, :, 2]

        else:
            picture = picture_open

    else:
        messagebox.showerror('Error!', 'Format file error!')
        return 0

    return picture, name_picture


def convert_csv_to_jpg():

    open_picture()
    
    if os.path.splitext(name_picture_open)[1] == '.csv':
        plt.imsave(name_picture_open + '.jpg', 100*picture, cmap='gray')
        messagebox.showinfo('Converting', 'Converting complete!')
    elif os.path.splitext(name_picture_open)[1] in file_extension:
        messagebox.showerror('Error!', 'Format file error!')


def define_delimiter(path):

    sniffer = csv.Sniffer()
    with open(path) as fp:
        delimiter = sniffer.sniff(fp.readline()).delimiter
    return delimiter


def is_number(str):

    try:
        float(str)
        return True

    except ValueError:
        return False


def exit_program():

    app.destroy()


if __name__ == "__main__":

    app = Interface()
    app.geometry('1280x720')
    app.title("Digital Holography")
    app.resizable(width=True, height=True)
    app.mainloop()
