import csv
import os
import os.path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math
import tkinter as tk
import tkinter.filedialog
import scipy.ndimage.filters

from numpy import genfromtxt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib import cm
from matplotlib.figure import Figure
from tkinter import messagebox, Frame
from PIL import ImageTk, Image
from skimage.restoration import unwrap_phase

matplotlib.use("TkAgg")


def doc_example():
    """Summary or Description of the Function

        Parameters:
        argument1 (int): Description of arg1

        Returns:
        int:Returning value
    """


def main_menu():
    """Summary or Description of the Function

        функция главного меню - выбора из списка функций
    """

    global frame1  # глобальные переменные - для передачи их в другие функции
    global frame2
    global frame3
    global frame4

    frame1: Frame = tk.Frame(window, width=1000, height=1000)  # первый фрейм с отображением главного меню
    frame2: Frame = tk.Frame(window)  # второй фрейм с отображением выбранной функции
    frame3: Frame = tk.Frame(window)  # В этом фрейме появляется картинка, график и т.д.
    frame4: Frame = tk.Frame(window)  # В этом фрейме появляется надпись, какой файл открыт

    frame1.place(x='20', y='20')  # положение фреймов в окне window по пикселям
    frame2.place(x='20', y='20')
    frame3.place(x='440', y='20')
    frame4.place(x='230', y='67')

    tk.Label(frame1, text="Enter a function:", font=("Calibri", 14)) \
        .grid(column=0, row=0, pady=0, padx=0)  # выводит на фрейм надпись

    # Тут и дальше: button - кнопка, нажатие которой выполняет соответствующую функцию,
    # указанную в аргументах. Метод grid размещает кнопку в "сетке"
    tk.Button(frame1, text="Fresnel reconstruction", font=("Calibri", 14), command=holography_reconstruction_fresnel) \
        .grid(column=0, row=2, pady=10, padx=25)

    tk.Button(frame1, text="Convolution reconstruction", font=("Calibri", 14)
              , command=holography_reconstruction_convolution) \
        .grid(column=0, row=4, pady=10, padx=25)

    tk.Button(frame1, text="Calculation", font=("Calibri", 14), command=synthetic_wavelength_function) \
        .grid(column=0, row=6, pady=10, padx=25)

    tk.Button(frame1, text="Convert CSV to JPG", font=("Calibri", 14), command=convert_csv_to_jpg) \
        .grid(column=0, row=8, pady=10, padx=25)

    tk.Button(frame1, text="Average picture", font=("Calibri", 14), command=average_picture) \
        .grid(column=0, row=10, pady=10, padx=25)

    tk.Button(frame1, text="Phase difference", font=("Calibri", 14), command=phase_difference_function) \
        .grid(column=0, row=12, pady=10, padx=25)

    tk.Button(frame1, text="3D-reconstruction", font=("Calibri", 14), command=topography) \
        .grid(column=0, row=14, pady=10, padx=25)

    tk.Button(frame1, text="Determination of recovery distance", font=("Calibri", 14), command=recovery_distance) \
        .grid(column=0, row=16, pady=10, padx=25)

    tk.Button(frame1, text="Hologram generation", font=("Calibri", 14), command=hologram_generation) \
        .grid(column=0, row=18, pady=10, padx=25)

    tk.Button(frame1, text="Exit", font=("Calibri", 14), command=window.destroy) \
        .grid(column=0, row=20, pady=10, padx=25)

    window.mainloop()  # метод, который не дает окну закрыться


def back_to_main_menu():  # подфункция возврата в главное меню

    frame2.destroy()  # фрейм 2 уничтожается, далее выполняется функция main, на котором фреймы обрисовываются заново

    if 'frame3' in globals():
        frame3.destroy()

    if 'frame4' in globals():
        frame4.destroy()

    main_menu()


def save_files():
    """Проверка файлов во всей папке,
    если выявляется файл с таким же названием,
    то запрашивается разрешение перезаписи
    """

    # проверка, что файла с такими названиями нет в папке:
    if os.path.exists(name_picture_text_input.get() + '_intensity.png')\
            or os.path.exists(name_picture_text_input.get() + '_phase.png')\
            or os.path.exists(name_picture_text_input.get() + '_intensity.csv')\
            or os.path.exists(name_picture_text_input.get() + '_phase.csv'):

        question = messagebox.askquestion("Warning!",
                                          "This name is already in use\nSave file?")  # выводит окно с переименованием

        if question == 'yes':
            save_image_and_csv_files()

    else:

        save_image_and_csv_files()


def save_image_and_csv_files():
    """Подфункция, которая сохраняет изображения и матрицы"""

    # сохранение изображения:
    plt.imsave(name_picture_text_input.get() + '_intensity.png', np.power(intensity, 0.4), cmap='gray')
    plt.imsave(name_picture_text_input.get() + '_phase.png', phase, cmap='gray')
    # Сохранение матрицы интенсивности:
    csv_file = open(name_picture_text_input.get() + '_intensity.csv', 'w')
    # построчная запись матрицы:
    with csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
        writer.writerows(intensity)
    # Сохранение матрицы фазы:
    csv_file = open(name_picture_text_input.get() + '_phase.csv', 'w')
    with csv_file:
        writer = csv.writer(csv_file, delimiter=';', lineterminator='\n')
        writer.writerows(phase)
    # сообщение на экране, что реконструкция прошла успешно:
    messagebox.showinfo('Reconstruction', 'Reconstruction complete!')

    frame3.place_forget()


def reconstruction_data_input():
    """Подфункция для ввода параметров цифрового голографического восстановления"""

    global pix_text_input
    global wav_text_input
    global z0_text_input

    # ввод текста в строку ввода:
    pix_text_input = tk.StringVar()
    wav_text_input = tk.StringVar()
    z0_text_input = tk.StringVar()

    '''Parameter input'''  # Ввод параметров во фрейм (с окном ввода)

    # Entry - добавление строки ввода (и ниже тоже):
    tk.Label(frame2, text='Wavelength (mm):').grid(column=0, row=6)
    tk.Entry(frame2, width=20, textvariable=wav_text_input).grid(column=1, row=6)

    tk.Label(frame2, text='Pixel size (mm):').grid(column=0, row=7)
    tk.Entry(frame2, width=20, textvariable=pix_text_input).grid(column=1, row=7)

    tk.Label(frame2, text='Reconstruction distance (mm):').grid(column=0, row=8)
    tk.Entry(frame2, width=20, textvariable=z0_text_input).grid(column=1, row=8)

    # кнопка выхода в главное меню:
    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu)\
        .grid(column=0, row=36, pady=10, padx=25)


def input_holography_reconstruction_fresnel():
    """Подфукнция для открытия голограммы, её обработки и ввода параметров для восстановления её методом Френеля"""

    global holo
    global name

    '''Open hologram'''

    holo = open_picture()
    # создаёт пустой фрейм, чтобы название файла на экране менялось корректно:
    tk.Label(frame4, text='                                   ').grid(column=0, row=0)

    holo_name = os.path.basename(name_picture_open)
    name = tk.Label(frame4, text=holo_name).grid(column=0, row=0)
    # Лапласиан (для устранения постоянной составляющей):
    holo = -scipy.ndimage.filters.laplace(holo)

    reconstruction_data_input()
    # запуск функции голографического восстановления методом Френеля:
    tk.Button(frame2, text="Enter", font=("Calibri", 14), command=click_holography_reconstruction_fresnel)\
        .grid(column=1, row=14, pady=10, padx=25)


def function_holography_reconstruction_fresnel():
    """Подфункция преобразования Френеля"""

    '''Reference beam'''  # Здесь задаётся функция опорного пучка (гаусс)

    N = len(holo)
    L = pix * N
    gradx_1 = np.zeros((N, N))
    gradx_sum = 0
    grady_1 = np.zeros((N, N))
    for j in range(len(holo)):
        gradx_sum += 1
        gradx_1[j] = (gradx_sum - N / 2) * pix
    grady_1 = np.rot90(gradx_1, k=1, axes=(0, 1))
    k = 2 * math.pi / wav
    refwav_1 = gradx_1 * gradx_1 + grady_1 * grady_1

    '''Fresnel transform'''  # Преобразование Френеля (преобразование Фурье + смещение спектра к центру)

    Fresnel = np.exp(1j * k / 2 / z0 * refwav_1)  # Коэффициент Френеля
    holo_fresnel = holo * Fresnel
    Fourier = np.fft.fft2(holo_fresnel)  # Преобразование Фурье
    Fourier = np.fft.fftshift(Fourier)  # Смещение нулевой частоты к центру спектра

    ipix = wav * np.abs(z0) / N / pix  # Размер пикселя при восстановлении
    gradx_2 = gradx_1 * ipix / pix
    grady_2 = grady_1 * ipix / pix
    refwav_2 = gradx_2 * gradx_2 + grady_2 * grady_2

    phase_coeff = np.exp(1j * k * z0) / (1j * wav * z0) * np.exp(1j * k / 2 / z0 * refwav_2)  # Коэффициент фазы

    global Reconstruction
    Reconstruction = Fourier * phase_coeff


def click_holography_reconstruction_fresnel():
    """Подфункция голографического восстановления методом Френеля"""

    global pix
    global wav
    global z0

    if is_number(pix_text_input.get()) == False or is_number(wav_text_input.get()) == False or is_number(
            z0_text_input.get()) == False:
        messagebox.showerror('Error!', 'You entered a non-numeric value!')

    else:

        pix = float(pix_text_input.get())
        wav = float(wav_text_input.get())
        z0 = float(z0_text_input.get())

        if pix == 0 or wav == 0 or z0 == 0:  # проверка на то, что все элементы - ненулевые
            messagebox.showerror('Error!', 'You entered zero value!')

        else:

            function_holography_reconstruction_fresnel()

            '''Intensity and phase matrix'''  # Определение сохраняемых данных

            global intensity
            global phase

            global image_holo_rec
            global intensity_picture

            intensity = np.power(np.abs(Reconstruction), 2)  # Интенсивность

            if np.mean(intensity) < 500000000:  # условие для адаптивной яркости изображения, отображаемого в frame3

                br = 0.3  # показатель степени для яркости

            else:

                br = 0.15

            image_holo_rec = Image.fromarray(np.power(intensity, br))
            image_holo_rec = image_holo_rec.resize((400, 400))

            intensity_picture = ImageTk.PhotoImage(image_holo_rec)

            tk.Label(frame3, image=intensity_picture).grid(column=0, row=0)

            # L0 = np.abs(wav*z0*N/L)    #ширина восстановленного изображения в миллиметрах
            # print('Image width = ',round(L0,3),'mm\n')
            # L0_text = Label(window, text = 'Image width = ',round(L0,3),'mm\n')
            # L0_text.grid(column = 0, row = 3)

            phase = np.arctan2(np.imag(Reconstruction), np.real(Reconstruction))  # Фаза
            #     phas = np.arctan(np.imag(U0)/np.real(U0))

            '''Save intensity and phase image'''  # Сохранение изображений и матриц

            # plt.gray()

            global name_picture_text_input

            name_picture_text_input = tk.StringVar()  # ввод имени файла

            tk.Label(frame2, text='Enter intensity and phase filename:').grid(column=0, row=20)
            tk.Entry(frame2, width=20, textvariable=name_picture_text_input).grid(column=1, row=20)

            tk.Button(frame2, text="OK", font=("Calibri", 14), command=save_files).grid(column=1, row=22, pady=10,
                                                                                        padx=25)  # вызов функции сохранения восстановленных голограмм


def holography_reconstruction_fresnel():  # общая функция реконструкции голограмм методом Френеля

    # функция ссылается на функцию открытия голограммы и ввода данных, которая в свою очередь ссылается на функцию восстановления голограммы

    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='Fresnel reconstruction', font=("Calibri", 14)).grid(column=0, row=0)
    tk.Button(frame2, text="Open hologram", font=("Calibri", 14), command=input_holography_reconstruction_fresnel).grid(
        column=0, row=2, pady=10, padx=25)

    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu).grid(column=0, row=36,
                                                                                                      pady=10, padx=25)


def input_holography_reconstruction_convolution():  # подфукнция для открытия голограммы, её обработки и ввода параметров для восстановления её методом свёртки

    global holo
    global name

    '''Open hologram'''

    holo = open_picture()

    tk.Label(frame4, text='                                   ').grid(column=0, row=0)
    holo_name = os.path.basename(name_picture_open)
    name = tk.Label(frame4, text=holo_name).grid(column=0, row=0)

    holo = -scipy.ndimage.filters.laplace(holo)  # Лапласиан (для устранения постоянной составляющей)

    reconstruction_data_input()

    tk.Button(frame2, text="Enter", font=("Calibri", 14), command=click_holography_reconstruction_convolution).grid(
        column=1, row=14, pady=10, padx=25)  # запуск функции голографического восстановления методом свёртки


def function_holography_reconstruction_convolution():  # подфункция преобразования свёрткой

    '''Reference beam'''  # Здесь задаётся функция опорного пучка (гаусс)

    N = len(holo)  # число пикселей голограммы по горизонтали
    L = pix * N  # ширина изображения в миллиметрах
    gradx_1 = np.zeros((N, N))
    gradx_sum = 0
    grady_1 = np.zeros((N, N))
    for i in range(len(holo)):
        gradx_sum += 1
        gradx_1[i] = (gradx_sum - N / 2) * pix
    grady_1 = np.rot90(gradx_1, k=1, axes=(0, 1))
    k = 2 * math.pi / wav
    refwav_1 = gradx_1 * gradx_1 + grady_1 * grady_1

    '''Convolution'''  # Метод свёртки (преобразование Фурье + смещение спектра к центру)

    weight = 1j * np.exp(-1j * k * np.sqrt(np.power(z0, 2) + np.power(gradx_1, 2) + np.power(grady_1, 2))) / (
            wav * np.sqrt(np.power(z0, 2) + np.power(gradx_1, 2) + np.power(grady_1, 2)))
    weight_f = np.fft.ifft2(weight)  # Обратное преобразование Фурье весовой функции
    weight_f = np.fft.ifftshift(weight_f)  # Смещение нулевой частоты весовой функции к центру спектра

    holo_f = np.fft.ifft2(holo)  # Обратное преобразование Фурье голограммы
    holo_f = np.fft.ifftshift(holo_f)  # Смещение нулевой частоты голограммы к центру спектра

    conv = np.fft.fft2(weight_f * holo_f)  # свёртка весовой функции и голограммы
    conv = np.fft.fftshift(conv)  # смещение нулевой частоты свёртки к центру спектра

    phase_coeff = np.exp(1j * k * z0) / (1j * wav * z0) * np.exp(1j * k / 2 / z0 * refwav_1)  # Коэффициент фазы

    global Reconstruction
    Reconstruction = conv * phase_coeff


def click_holography_reconstruction_convolution():  # подфункция голографического восстановления методом свёртки

    global pix
    global wav
    global z0

    if is_number(pix_text_input.get()) == False or is_number(wav_text_input.get()) == False or is_number(
            z0_text_input.get()) == False:
        messagebox.showerror('Error!', 'You entered a non-numeric value!')

    else:

        pix = float(pix_text_input.get())
        wav = float(wav_text_input.get())
        z0 = float(z0_text_input.get())

        if pix == 0 or wav == 0 or z0 == 0:  # проверка на то, что все элементы - ненулевые
            messagebox.showerror('Error!', 'You entered zero value!')

        else:

            function_holography_reconstruction_convolution()

            '''Intensity and phase matrix'''  # Определение сохраняемых данных

            global intensity
            global phase

            global image_holo_rec
            global intensity_picture

            intensity = np.power(np.abs(Reconstruction), 2)  # Интенсивность

            image_holo_rec = Image.fromarray(np.power(intensity, 0.8))
            image_holo_rec = image_holo_rec.resize((400, 400))

            intensity_picture = ImageTk.PhotoImage(image_holo_rec)

            tk.Label(frame3, image=intensity_picture).grid(column=0, row=0)

            # L0 = np.abs(wav*z0*N/L)
            # print('Image width = ',round(L0,3),'mm') 

            phase = np.arctan2(np.imag(Reconstruction), np.real(Reconstruction))  # Фаза
            #     phas = np.arctan(np.imag(U0)/np.real(U0))

            '''Save intensity and phase image'''  # Сохранение изображений и матриц

            # plt.gray()

            global name_picture_text_input

            name_picture_text_input = tk.StringVar()

            tk.Label(frame2, text='Enter intensity and phase filename:').grid(column=0, row=20)
            tk.Entry(frame2, width=20, textvariable=name_picture_text_input).grid(column=1, row=20)

            tk.Button(frame2, text="OK", font=("Calibri", 14), command=save_files).grid(column=1, row=22, pady=10,
                                                                                        padx=25)  # вызов функции сохранения восстановленных голограмм


def holography_reconstruction_convolution():  # общая функция реконструкции голограмм методом Френеля

    # функция ссылается на функцию открытия голограммы и ввода данных, которая в свою очередь ссылается на функцию восстановления голограммы

    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='Convolution reconstruction', font=("Calibri", 14)).grid(column=0, row=0)
    tk.Button(frame2, text="Open hologram", font=("Calibri", 14),
              command=input_holography_reconstruction_convolution).grid(column=0, row=2, pady=10, padx=25)

    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu).grid(column=0, row=36,
                                                                                                      pady=10, padx=25)


def click_synthetic_wavelength_function():  # подфункция подсчёта длины волны биений, максимальную длину восстановления, ширину восстановленного изображения в миллиметрах

    if is_number(wav_1_text_input.get()) == False or is_number(wav_2_text_input.get()) == False or is_number(
            pix_text_input.get()) == False or is_number(z0_text_input.get()) == False:
        messagebox.showerror('Error!', 'You entered a non-numeric value!')

    else:

        wav_1 = float(wav_1_text_input.get())
        wav_2 = float(wav_2_text_input.get())
        pix = float(pix_text_input.get())
        z0 = float(z0_text_input.get())

        if pix == 0 or wav_1 == 0 or wav_2 == 0 or z0 == 0:  # проверка на нулевые значения
            messagebox.showerror('Error!', 'You entered zero value!')

        else:

            if wav_1 == wav_2:  # Если две длины волны равны, то биений не будет
                messagebox.showerror('Error!', 'You entered same wavelenght!')


            else:  # если все значения ненулевые
                L0 = np.abs(wav_1 * z0 / pix)
                synt_wav = np.abs(wav_1 * wav_2 / (wav_1 - wav_2))  # формула нахождения длины волны биений

                L0_text = str(
                    round(L0, 3))  # считывает обработанные переменные и преобразует их в текстовые данные с округлением
                synt_wav_text = str(round(synt_wav, 4))
                rough_depth_text = str(round(synt_wav / 2, 4))

                L0_string = 'Image width = ' + L0_text + 'mm'
                synt_wav_string = 'Syntethic wave = ' + synt_wav_text + 'mm'
                rough_depth_string = 'Roughness depth = ' + rough_depth_text + 'mm'

                tk.Label(frame2, text=synt_wav_string).grid(column=0, row=20)

                tk.Label(frame2, text=L0_string).grid(column=0, row=24)

                tk.Label(frame2, text=rough_depth_string).grid(column=0, row=28)


def multiwavelength_data_input():  # ввод данных для функции вычисления длины волны биений (нужна ещё и для функции построения профиля поверхности)

    reconstruction_data_input()  # задание параметров из вышеописанной функции

    global wav_1_text_input
    global wav_2_text_input

    wav_1_text_input = tk.StringVar()
    wav_2_text_input = tk.StringVar()

    tk.Label(frame2, text='Wavelength 1 (mm): ').grid(column=0,
                                                      row=4)  # новый параметр, перекрывающий изначальный из функции reconstruction_data_input()
    tk.Entry(frame2, width=20, textvariable=wav_1_text_input).grid(column=1, row=4)

    tk.Label(frame2, text='Wavelength 2 (mm): ').grid(column=0, row=6)  # новый добавленный параметр()
    tk.Entry(frame2, width=20, textvariable=wav_2_text_input).grid(column=1, row=6)


def synthetic_wavelength_function():  # Функция вычисления длины волны биений двух длин волн

    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='Synthetic wavelength calculate', font=("Calibri", 14)).grid(column=0, row=0)

    multiwavelength_data_input()

    tk.Button(frame2, text="Enter", font=("Calibri", 14), command=click_synthetic_wavelength_function).grid(column=1,
                                                                                                            row=14,
                                                                                                            pady=10,
                                                                                                            padx=25)


def csv_average_picture():  # подфункция усреднения нескольких голограмм в формате *.csv

    dir = tk.filedialog.askdirectory()  # выбор директории

    all_picture = [f for f in os.listdir(dir) if f.endswith('.csv')]  # массив с матрицами в формате *.csv

    count = 0

    if all_picture == []:  # если файлов нет, вывести ошибку

        messagebox.showerror('Error!', 'No such files!')

    else:

        N = len(genfromtxt(all_picture[0], delimiter=';'))  # число пикселей голограммы по горизонтали

        for i in range(len(all_picture)):

            N_local = len(genfromtxt(all_picture[i],
                                     delimiter=';'))  # определение числа пикселей отдельного изображения в массиве изображений

            if N_local != N:  # если есть изображения, которые отличаются по размеру, то выдать ошибку
                messagebox.showerror('Error!', 'Images have different resolutions')

        sum = np.zeros((N, N))

        for i in range(len(all_picture)):
            picture = genfromtxt(all_picture[i], delimiter=';')
            N = len(picture)
            picture = picture[:, :N]  # обрезка изображения до квадратного вида
            sum += picture
            count += 1

        average_picture = sum / count  # среднее арифметическое изображений

        plt.imsave('average.png', average_picture, cmap='gray')

        CSV_file = open('average.csv', 'w')
        with CSV_file:
            writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
            writer.writerows(average_picture)

        messagebox.showinfo('Averaging',
                            'Averaging complete!')  # вывод информационного сообщения о завершении усреднения


def image_average_picture():  # подфункция усреднения нескольких голограмм в формате изображений

    dir = tk.filedialog.askdirectory()  # выбор директории

    all_picture = [f for f in os.listdir(dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith(
        '.bmp')]  # массив с матрицами в формате png, jpg или bmp

    count = 0

    if all_picture == []:

        messagebox.showerror('Error!', 'No such files!')

    else:

        N = len(img.imread(all_picture[0]))  # число пикселей голограммы по горизонтали

        for i in range(len(all_picture)):

            N_local = len(
                img.imread(all_picture[i]))  # определение числа пикселей отдельного изображения в массиве изображений

            if N_local != N:  # если есть изображения, которые отличаются по размеру, то выдать ошибку
                messagebox.showerror('Error!', 'Images have different resolutions')

        sum = np.zeros((N, N))
        picture = np.zeros((N, N))
        picture_1 = np.zeros((N, N))

        for i in range(len(all_picture)):
            picture = img.imread(all_picture[i])
            N = len(picture)
            np.array(picture[:, :N],
                     dtype=float)  # обрезка изображений до квадрата и преобразование значений интенсивности в тип float
            if picture.ndim == 3:
                picture_1 = picture[:, :,
                            0]  # выделение одного цветового канала, если изображение имеет три цветовых канала
            else:
                picture_1 = picture  # если цветовой канал 1, то изображение не меняется
            sum += picture_1
            count += 1

        average_picture = sum / count  # среднее арифметическое изображений

        plt.imsave('average.png', average_picture, cmap='gray')

        CSV_file = open('average.csv', 'w')
        with CSV_file:
            writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
            writer.writerows(average_picture)

        messagebox.showinfo('Averaging',
                            'Averaging complete!')  # вывод информационного сообщения о завершении усреднения


def average_picture():  # Функция усредения всех картинок или матриц, находящихся в папке (выбор типа изображений по кнопке)

    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='Average picture', font=("Calibri", 14)).grid(column=0, row=0)

    tk.Button(frame2, text='CSV-files', font=("Calibri", 14), command=csv_average_picture).grid(column=0, row=2,
                                                                                                pady=10, padx=25)

    tk.Button(frame2, text='Image files', font=("Calibri", 14), command=image_average_picture).grid(column=0, row=4,
                                                                                                    pady=10, padx=25)

    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu).grid(column=0, row=36,
                                                                                                      pady=10, padx=25)


def phase_1_open():  # открытие первой фазы

    global phase_1
    phase_1 = open_picture()

    phase_1_name = os.path.basename(name_picture_open)

    tk.Label(frame2, text='                                   ').grid(column=1, row=2)
    tk.Label(frame2, text=phase_1_name).grid(column=1, row=2)


def phase_2_open():  # открытие первой фазы

    global phase_2
    phase_2 = open_picture()

    phase_2_name = os.path.basename(name_picture_open)

    global name
    name = tk.Label(frame2, text='                                   ').grid(column=1, row=4)
    tk.Label(frame2, text=phase_2_name).grid(column=1, row=4)


def filter_phase_diff():  # подфункция фильтрации разности фаз

    phase_diff_filtered = scipy.ndimage.median_filter(phase_diff, size=(1, 1))  # медианный фильтр

    for k in range(30):  # алгоритм sin-cos фильтрации
        sin_phase_diff = np.sin(phase_diff_filtered)
        cos_phase_diff = np.cos(phase_diff_filtered)
        sin_phase_diff_filtered = scipy.ndimage.median_filter(sin_phase_diff, size=(5, 5))
        cos_phase_diff_filtered = scipy.ndimage.median_filter(cos_phase_diff, size=(5, 5))
        phase_diff_filtered = np.arctan2(cos_phase_diff_filtered, sin_phase_diff_filtered)

    plt.imsave(name_phase_diff_text_input.get() + '_filtered.png', phase_diff_filtered, cmap='gray')

    CSV_file = open(name_phase_diff_text_input.get() + '_filtered.csv', 'w')
    with CSV_file:
        writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
        writer.writerows(phase_diff_filtered)

    phase_diff_unwrapped = unwrap_phase(phase_diff_filtered)  # функция развертки фазы

    plt.imsave(name_phase_diff_text_input.get() + '_unwrapped_filtered.png', phase_diff_unwrapped, cmap='gray')

    CSV_file = open(name_phase_diff_text_input.get() + '_unwrapped_filtered.csv', 'w')
    with CSV_file:
        writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
        writer.writerows(phase_diff_unwrapped)


def save_phase_diff():  # Проверка файлов во всей папке, если выявляется файл с таким же названием, то запрашивается разрешение перезаписи

    if os.path.exists(name_phase_diff_text_input.get() + '.csv') or os.path.exists(
            name_phase_diff_text_input.get() + '_filtered.csv') or os.path.exists(
        name_phase_diff_text_input.get() + '__unwrapped_filtered.csv'):  # проверка, что файла с такими названиями нет в папке

        question = messagebox.askquestion("Warning!",
                                          "This name is already in use\nSave file?")  # выводит окно с переименованием

        if question == 'yes':
            save_image_and_csv_phase_diff()

    else:

        save_image_and_csv_phase_diff()


def save_image_and_csv_phase_diff():  # присвоение имени файла разности фаз и сохранеие в формате изображения и матрицы *.csv

    plt.imsave(name_phase_diff_text_input.get() + '.png', phase_diff, cmap='gray')

    CSV_file = open(name_phase_diff_text_input.get() + '.csv', 'w')
    with CSV_file:
        writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
        writer.writerows(phase_diff)

    filter_phase_diff()

    messagebox.showinfo('Save phase difference',
                        'Phase difference complete!')  # вывод информационного сообщения о завершении фильтрации


def click_phase_difference_function():  # подфункция подсчёта разности фаз

    global phase_diff

    phase_diff = np.zeros((len(phase_1), len(phase_1)))

    for i in range(len(phase_1)):  # разность фаз
        for j in range(len(phase_1[i])):
            if phase_2[i][j] - phase_1[i][j] < 0:
                phase_diff[i][j] = phase_2[i][j] - phase_1[i][j] + 2 * math.pi
            else:
                phase_diff[i][j] = phase_2[i][j] - phase_1[i][j]

    global name_phase_diff_text_input

    name_phase_diff_text_input = tk.StringVar()  # присвоение имени файла разности фаз

    tk.Label(frame2, text='Enter phase difference filename:').grid(column=0, row=20)
    tk.Entry(frame2, width=20, textvariable=name_phase_diff_text_input).grid(column=1, row=20)

    tk.Button(frame2, text="OK", font=("Calibri", 14), command=save_phase_diff).grid(column=1, row=22, pady=10,
                                                                                     padx=25)  # кнопка, вызывающая функцию сохранения разности фаз в файл


def phase_difference_function():  # Функция подсчёта разности фаз двух состояний

    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='Phase difference', font=("Calibri", 14)).grid(column=0, row=0)

    tk.Button(frame2, text='Phase 1', font=("Calibri", 14), command=phase_1_open).grid(column=0, row=2, pady=10,
                                                                                       padx=25)

    tk.Button(frame2, text='Phase 2', font=("Calibri", 14), command=phase_2_open).grid(column=0, row=4, pady=10,
                                                                                       padx=25)

    tk.Button(frame2, text="Enter", font=("Calibri", 14), command=click_phase_difference_function).grid(column=0, row=8,
                                                                                                        pady=10,
                                                                                                        padx=25)  # кнопка запуска подфункции подсчета разности фаз

    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu).grid(column=0, row=36,
                                                                                                      pady=10, padx=25)


def calculate_topography_function():  # подфункция вычисления трёхмерного профиля поверхности

    if is_number(wav_1_text_input.get()) == False or is_number(wav_2_text_input.get()) == False or is_number(
            pix_text_input.get()) == False or is_number(z0_text_input.get()) == False:
        messagebox.showerror('Error!', 'You entered a non-numeric value!')

    else:

        wav_1 = float(wav_1_text_input.get())  # считывание данных со строки с переводом в тип float
        wav_2 = float(wav_2_text_input.get())
        pix = float(pix_text_input.get())
        z0 = float(z0_text_input.get())

        if wav_1 == wav_2:  # условие, что длины волн разные
            messagebox.showerror('Error!', 'You entered same wavelenght!')

        if pix == 0 or wav_1 == 0 or wav_2 == 0 or z0 == 0:  # условие, что нет значений, равных 0
            messagebox.showerror('Error!', 'You entered zero value!')

        synt_wav = np.abs(wav_1 * wav_2 / (wav_1 - wav_2))  # формула нахождения длины волны биений

        global depth_map

        depth_map = synt_wav / (4 * math.pi) * topography_picture  # карта высот поверхности

        global N
        global x, y
        global L0

        N = len(depth_map)
        x, y = np.meshgrid(np.arange(N),
                           np.arange(N))  # задание линейной функции возрастания для двух других осей объекта

        L0 = np.abs(wav_1 * z0 / pix)  # ширина объекта

        # # Цикл для того, чтобы убрать слишком пересвеченные части, обрезка их, обнуление
        # for i in range(N):
        #     for j in range(N):
        #         if phase_diff[i][j] > 0.006:
        #             phase_diff[i][j] = 0


def frame_topography():
    # Отрисовка поверхности на фрейме 3

    calculate_topography_function()

    fig = Figure(figsize=(5, 4), dpi=100)  # параметры окна для графика
    canvas = FigureCanvasTkAgg(fig, frame3)  # задаёт холст для графика

    canvas.draw()  # отрисовка холста

    canvas.get_tk_widget().grid(column=1, row=0)  # положение холста

    axes = fig.add_subplot(111, projection='3d')  # задаёт оси графика

    axes.set_xlabel('Width,mm', fontsize=12)  # подписи осей графика
    axes.set_ylabel('Width,mm', fontsize=12)
    axes.set_zlabel('Depth,mm', fontsize=12)

    topography_image = axes.plot_surface(x * L0 / N, y * L0 / N, depth_map, cmap=cm.coolwarm, linewidth=0,
                                         antialiased=False)  # трёхмерная визуализация поверхности


def new_window_topography():  # Вывод графика в отдельном окне

    calculate_topography_function()

    fig = Figure(figsize=(5, 4), dpi=175)

    plt.rcParams['font.size'] = '12'

    axes = plt.subplot(111, projection='3d')

    axes.set_xlabel('Width,mm', fontsize=12)  # подписи осей графика
    axes.set_ylabel('Width,mm', fontsize=12)
    axes.set_zlabel('Depth,mm', fontsize=12)

    topography_image = axes.plot_surface(x * L0 / N, y * L0 / N, depth_map, cmap=cm.coolwarm, linewidth=0,
                                         antialiased=False)  # трёхмерная визуализация поверхности

    plt.show()


def input_topography_function():
    global topography_picture

    topography_picture = open_picture()

    tk.Label(frame4, text='                                   ').grid(column=0, row=0)
    topography_name = os.path.basename(name_picture_open)
    tk.Label(frame4, text=topography_name).grid(column=0, row=0)

    multiwavelength_data_input()  # функция, которая вызывает ввод параметров для определения высоты профиля поверхности матрицы

    tk.Button(frame2, text="Window interface", font=("Calibri", 14), command=new_window_topography).grid(column=0,
                                                                                                         row=12,
                                                                                                         pady=10,
                                                                                                         padx=25)  # кнопка для выполнения фукнции построения трёхмерного профиля поверхности в отдельном окне

    tk.Button(frame2, text="TKinter interface", font=("Calibri", 14), command=frame_topography).grid(column=1, row=12,
                                                                                                     pady=10,
                                                                                                     padx=25)  # кнопка для выполнения фукнции построения трёхмерного профиля поверхности в текущем фрейме


def topography():  # Функция построения топограммы поверхности

    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='3D-reconstruction', font=("Calibri", 14)).grid(column=0, row=0)

    tk.Button(frame2, text="Open height map", font=("Calibri", 14), command=input_topography_function).grid(column=0,
                                                                                                            row=2,
                                                                                                            pady=10,
                                                                                                            padx=25)

    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu).grid(column=0, row=36,
                                                                                                      pady=10, padx=25)


def open_picture():  # Функция открытия изображения

    global name_picture_open
    #     root = tk.Tk()
    #     root.update()
    # name_picture_open = tk.filedialog.askopenfilename(initialdir=r"D:/COIL/Holo/DHII/")

    name_picture_open = tk.filedialog.askopenfilename()  # вызываем диалог открытия файла
    #     root.destroy()

    global picture

    global name_picture
    name_picture = os.path.splitext(name_picture_open)[0]

    if os.path.splitext(name_picture_open)[
        1] == '.csv':  # делим функцией os.path.splitext() название файла на имя и расширение, и сравниваем расширение с указанным
        picture_open = genfromtxt(name_picture_open, delimiter=';')  # преобразуем csv в двумерный массив
        N = len(picture_open)  # высота картинки в пикселях
        picture = picture_open[:,
                  :N]  # обрезаем ширину картинки до высоты (если высота больше ширины, то произойдет ошибка)

    elif os.path.splitext(name_picture_open)[1] == '.jpg' or os.path.splitext(name_picture_open)[1] == '.JPG' or \
            os.path.splitext(name_picture_open)[1] == '.png' or os.path.splitext(name_picture_open)[1] == '.bmp':
        picture_open = img.imread(name_picture_open)  # преобразуем картинку в массив
        if np.shape(picture_open)[0] > np.shape(picture_open)[1]:
            picture_open = np.rot90(picture_open, k=1, axes=(0, 1))
        N = len(picture_open)
        picture_open = np.array(picture_open[:, :N], dtype=float)

        if picture_open.ndim == 3:  # если размерность массива 3, то используем один канал
            picture = picture_open[:, :, 0]

        else:
            picture = picture_open

    else:
        print('Format file error!')

    return picture


def convert_csv_to_jpg():
    picture = open_picture()
    plt.imsave(name_picture + '.jpg', picture, cmap='gray')


def is_number(str):
    try:
        float(str)
        return True

    except ValueError:
        return False


def input_recovery_distance():
    global holo

    '''Open hologram'''

    holo = open_picture()

    holo_name = os.path.basename(name_picture_open)
    tk.Label(frame4, text=holo_name).grid(column=0, row=0)

    holo = -scipy.ndimage.filters.laplace(holo)  # Лапласиан (для устранения постоянной составляющей)

    reconstruction_data_input()

    global step_text_input
    global min_z0_text_input
    global max_z0_text_input

    step_text_input = tk.StringVar()
    min_z0_text_input = tk.StringVar()
    max_z0_text_input = tk.StringVar()

    tk.Label(frame2, text='Min reconstruction distance (mm):').grid(column=0, row=8)
    tk.Entry(frame2, width=20, textvariable=min_z0_text_input).grid(column=1, row=8)

    tk.Label(frame2, text='Max reconstruction distance (mm):').grid(column=0, row=10)
    tk.Entry(frame2, width=20, textvariable=max_z0_text_input).grid(column=1, row=10)

    tk.Label(frame2, text='Step distance reconstruction (mm):').grid(column=0, row=12)
    tk.Entry(frame2, width=20, textvariable=step_text_input).grid(column=1, row=12)

    tk.Button(frame2, text="Enter", font=("Calibri", 14), command=click_recovery_distance).grid(column=1, row=14,
                                                                                                pady=10,
                                                                                                padx=25)  # запуск функции голографического восстановления методом Френеля


def click_recovery_distance():
    global pix
    global wav
    global min_z0
    global max_z0
    global step

    if is_number(pix_text_input.get()) == False or is_number(wav_text_input.get()) == False or is_number(
            min_z0_text_input.get()) == False or is_number(max_z0_text_input.get()) == False or is_number(
        step_text_input.get()) == False:
        messagebox.showerror('Error!', 'You entered a non-numeric value!')

    else:

        pix = float(pix_text_input.get())
        wav = float(wav_text_input.get())
        min_z0 = float(min_z0_text_input.get())  # начальная дистанция восстановления
        max_z0 = float(max_z0_text_input.get())  # конечная дистанция восстановления
        step = float(step_text_input.get())

        if pix == 0 or wav == 0 or max_z0 == 0 or min_z0 == 0 or step == 0:  # проверка на то, что все элементы - ненулевые
            messagebox.showerror('Error!', 'You entered zero value!')

        else:

            global z0

            z0 = min_z0  # счётчик дистанции восстановления

            global array_intensity

            array_intensity = np.zeros((round((
                                                      max_z0 - min_z0) / step) + 1))  # нулевой массив с числом элементов, равным количеству шагов по дистанции

            question = messagebox.askquestion("Attention",
                                              "Save reconstruction pictures?")  # выводит окно с переименованием

            for i in range(len(array_intensity)):  # цикл восстановления с разными шагами

                # '''Loading string'''

                # bar = ttk.Progressbar(frame2, length = 200, mode="determinate", variable = z0, maximum = max_z0).start()  

                # bar['value'] = z0/100
                # bar.grid(column = 0, row = 20)  

                function_holography_reconstruction_fresnel()

                '''Intensity and phase matrix'''  # Определение сохраняемых данных

                global intensity

                intensity = np.power(np.abs(Reconstruction), 2)  # Интенсивность

                '''Save intensity images'''  # Сохранение изображений

                if question == 'yes':
                    plt.imsave(str(z0) + '_intensity.png', np.power(intensity, 0.4), cmap='gray')

                '''Define average intensity'''

                intensity = intensity / np.max(intensity)

                sum_intensity = np.sum(
                    np.sum(intensity))  # подсчёт суммы интенсивностей всех пикселей восстановленного изображения
                array_intensity[i] = sum_intensity  # заполнение массива с суммами

                '''Next step'''

                z0 = z0 + step  # следующий шаг

            '''Determinate distance'''

            array_intensity = array_intensity.tolist()  # преобразование numpy.array в list

            index_min_intensity = array_intensity.index(min(array_intensity))  # индекс минимального элемента массива
            z0_corr_reconsctruction = min_z0 + index_min_intensity * step
            z0_corr_reconsctruction_text = str(z0_corr_reconsctruction)
            z0_corr_reconsctruction_string = 'Recovery distance (mm): ' + z0_corr_reconsctruction_text + 'mm'
            tk.Label(frame2, text=z0_corr_reconsctruction_string).grid(column=0, row=16)

            array_intensity[index_min_intensity] = max(
                array_intensity)  # теперь минимальному элементу массива присваивается значение, например, максимального элемента, чтобы найти второй с конца минимальный элемент
            index_min_intensity = array_intensity.index(min(array_intensity))
            z0_corr_reconsctruction = min_z0 + index_min_intensity * step
            z0_corr_reconsctruction_text = str(z0_corr_reconsctruction)
            z0_corr_reconsctruction_string = 'Extra distance recovery (mm): ' + z0_corr_reconsctruction_text + 'mm'
            tk.Label(frame2, text=z0_corr_reconsctruction_string).grid(column=0, row=18)


def recovery_distance():  # фунцкия восстановления голограмм с определённым шагом по дистанции, а также позволяющая по наиболее затемнённому изображению определять дистанцию восстановления

    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='Recovery distance', font=("Calibri", 14)).grid(column=0, row=0)

    tk.Button(frame2, text="Open hologram", font=("Calibri", 14), command=input_recovery_distance).grid(column=0, row=2,
                                                                                                        pady=10,
                                                                                                        padx=25)

    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu).grid(column=0, row=36,
                                                                                                      pady=10, padx=25)


def hologram_generation():
    frame1.destroy()  # метод для очистки фрейма 1 с главным меню, после чего отрисовывается фрейм 2

    tk.Label(frame2, text='Hologram generation', font=("Calibri", 14)).grid(column=0, row=0)

    tk.Button(frame2, text="Open picture", font=("Calibri", 14), command=input_hologram_generation).grid(column=0,
                                                                                                         row=2, pady=10,
                                                                                                         padx=25)

    tk.Button(frame2, text="Back to main menu", font=("Calibri", 14), command=back_to_main_menu).grid(column=0, row=36,
                                                                                                      pady=10, padx=25)


def input_hologram_generation():
    global picture_initial
    global name

    picture_initial = open_picture()

    tk.Label(frame4, text='                                   ').grid(column=0, row=0)
    picture_initial_name = os.path.basename(name_picture_open)
    tk.Label(frame4, text=picture_initial_name).grid(column=0, row=0)

    reconstruction_data_input()

    global size_text_input
    size_text_input = tk.StringVar()

    tk.Label(frame2, text='Object size (mm):').grid(column=0, row=7)
    tk.Entry(frame2, width=20, textvariable=size_text_input).grid(column=1, row=7)

    tk.Button(frame2, text="Enter", font=("Calibri", 14), command=click_hologram_generation).grid(column=1, row=14,
                                                                                                  pady=10,
                                                                                                  padx=25)  # запуск функции голографического восстановления методом Френеля


def save_files_hologram():
    if os.path.exists('hologram_' + name_hologram_text_input.get() + '.png') or os.path.exists(
            'hologram_' + name_hologram_text_input.get() + '.csv'):  # проверка, что файла с такими названиями нет в папке

        question = messagebox.askquestion("Warning!",
                                          "This name is already in use\nSave file?")  # выводит окно с переименованием

        if question == 'yes':
            save_image_and_csv_hologram()

    else:

        save_image_and_csv_hologram()


def save_image_and_csv_hologram():
    plt.imsave('hologram_' + name_hologram_text_input.get() + '.png', hologram[size_picture:, size_picture:],
               cmap='gray')

    CSV_file = open('hologram_' + name_hologram_text_input.get() + '.csv', 'w')  # Сохранение матрицы фазы
    with CSV_file:
        writer = csv.writer(CSV_file, delimiter=';', lineterminator='\n')
        writer.writerows(hologram[size_picture:, size_picture:])

    messagebox.showinfo('Hologram generation',
                        'Hologram generation complete!')  # сообщение на экране, что реконструкция прошла успешно


def click_hologram_generation():
    if is_number(wav_text_input.get()) == False or is_number(z0_text_input.get()) == False or is_number(
            size_text_input.get()) == False:
        messagebox.showerror('Error!', 'You entered a non-numeric value!')

    else:

        wav = float(wav_text_input.get())
        z0 = float(z0_text_input.get())
        size = float(size_text_input.get())

        if wav == 0 or z0 == 0 or size == 0:  # проверка на то, что все элементы - ненулевые
            messagebox.showerror('Error!', 'You entered zero value!')

        else:

            global size_picture
            size_picture = np.size(picture_initial, 0)
            size_matrix = 2 * size_picture

            global object_holo
            global zeros_1
            global zeros_2
            global object_holo_aver

            k = 2 * math.pi / wav
            zeros_1 = np.zeros((size_matrix, (size_matrix - size_picture) // 2))
            zeros_2 = np.zeros(((size_matrix - size_picture) // 2, size_picture))

            object_holo_aver = np.vstack((np.vstack((zeros_2, picture_initial)), zeros_2))
            object_holo = np.hstack((np.hstack((zeros_1, object_holo_aver)), zeros_1))

            pix = np.abs(z0) * wav / size
            size_x = pix * size_matrix
            size_y = pix * size_matrix

            phase_rand = 2 * math.pi * (np.random.rand(size_matrix, size_matrix) - 0.5)
            complex_obj = object_holo * np.exp(1j * phase_rand)

            gradx_1 = np.zeros((size_matrix, size_matrix))
            gradx_sum = 0
            grady_1 = np.zeros((size_matrix, size_matrix))
            for j in range(len(picture_initial)):
                gradx_sum += 1
                gradx_1[j] = (gradx_sum - size_matrix / 2) * pix
            grady_1 = np.rot90(gradx_1, k=1, axes=(0, 1))
            k = 2 * math.pi / wav
            refwav_1 = gradx_1 * gradx_1 + grady_1 * grady_1

            Fresnel = np.exp(1j * k / 2 / z0 * refwav_1)
            obj_fresnel = complex_obj * Fresnel

            Fourier = np.fft.fft2(obj_fresnel)  # Преобразование Фурье
            Fourier = np.fft.fftshift(Fourier)  # Смещение нулевой частоты к центру спектра

            ipix = wav * np.abs(z0) / size_matrix / pix  # Размер пикселя при восстановлении
            gradx_2 = gradx_1 * ipix / pix
            grady_2 = grady_1 * ipix / pix
            refwav_2 = gradx_2 * gradx_2 + grady_2 * grady_2

            phase_coeff = np.exp(1j * k * z0) / (1j * wav * z0) * np.exp(1j * k / 2 / z0 * refwav_2)  # Коэффициент фазы
            hologram_before = Fourier * phase_coeff
            freq = size_x / 8 / wav / z0
            amplitude = np.max(np.max(np.abs(hologram_before)))

            ref_wave = amplitude * np.exp(2 * 1j * math.pi * freq * (gradx_1 + gradx_2))
            hologram_med = np.power(np.abs(ref_wave + hologram_before), 2)
            max_hologram_obj = np.max(np.max(hologram_med))

            global hologram
            hologram = 255 * hologram_med / max_hologram_obj

            global name_hologram_text_input
            name_hologram_text_input = tk.StringVar()  # ввод имени файла

            tk.Label(frame2, text='Enter hologram filename:').grid(column=0, row=20)
            tk.Entry(frame2, width=20, textvariable=name_hologram_text_input).grid(column=1, row=20)
            # вызов функции сохранения восстановленных голограмм:
            tk.Button(frame2, text="OK", font=("Calibri", 14), command=save_files_hologram)\
                .grid(column=1, row=22, pady=10, padx=25)


window = tk.Tk()  # открытие главного окна

window.geometry('1000x1000')  # размер главного окна
window.title("Digital Holography")

window.resizable(width=True, height=True)  # возможность изменения размера окна

main_menu()  # главное меню
