import random
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tabulate import tabulate


class Node:
    """Вычислительный узел кластера"""
    def __init__(self, node_id, max_cpu, max_ram):
        self.id = node_id
        self.max_cpu = max_cpu          # Максимальный CPU в ядрах
        self.max_ram = max_ram          # Максимальная RAM в МБ
        self.used_cpu = 0               # Использованный CPU
        self.used_ram = 0                # Использованная RAM
        self.containers = []             # Размещенные контейнеры
        
    def reset(self):
        """Сброс узла в начальное состояние"""
        self.used_cpu = 0
        self.used_ram = 0
        self.containers = []
        
    def can_fit(self, container):
        """Проверка, помещается ли контейнер"""
        return (self.used_cpu + container.cpu <= self.max_cpu and 
                self.used_ram + container.ram <= self.max_ram)
    
    def add_container(self, container):
        """Добавление контейнера на узел"""
        if self.can_fit(container):
            self.used_cpu += container.cpu
            self.used_ram += container.ram
            self.containers.append(container)
            container.node_id = self.id
            container.placed = True
            return True
        return False
    
    def get_load_cpu(self):
        """Загрузка CPU в процентах"""
        return (self.used_cpu / self.max_cpu) * 100 if self.max_cpu > 0 else 0
    
    def get_load_ram(self):
        """Загрузка RAM в процентах"""
        return (self.used_ram / self.max_ram) * 100 if self.max_ram > 0 else 0
    
    def __repr__(self):
        return f"Node(id={self.id}, CPU={self.used_cpu}/{self.max_cpu}, RAM={self.used_ram}/{self.max_ram})"


class Container:
    """Контейнер приложения"""
    def __init__(self, cont_id, cpu, ram):
        self.id = cont_id
        self.cpu = cpu                   # Требуемый CPU в ядрах
        self.ram = ram                   # Требуемая RAM в МБ
        self.node_id = 0                  # Узел размещения (0 - не размещен)
        self.placed = False                # Статус размещения
        
    def reset(self):
        """Сброс контейнера в начальное состояние"""
        self.node_id = 0
        self.placed = False
        
    def __repr__(self):
        status = "размещен" if self.placed else "не размещен"
        return f"Container(id={self.id}, CPU={self.cpu}, RAM={self.ram}, {status})"


def initialize_cluster(nodes_data):
    """
    Инициализация кластера
    F₁ = InitializeCluster(N)
    """
    nodes = []
    for i, (cpu, ram) in enumerate(nodes_data, 1):
        nodes.append(Node(i, cpu, ram))
    return nodes


def initialize_containers(containers_data):
    """
    Инициализация контейнеров
    F₂ = InitializeContainers(C)
    """
    containers = []
    for i, (cpu, ram) in enumerate(containers_data, 1):
        containers.append(Container(i, cpu, ram))
    return containers


def reset_system(nodes, containers):
    """Сброс системы в начальное состояние"""
    for node in nodes:
        node.reset()
    for container in containers:
        container.reset()



def first_fit_placement(containers, nodes):
    """
    Метод First Fit (z = 1)
    Каждый контейнер размещается на первом подходящем узле
    """
    placement = [0] * len(containers)
    
    for container in containers:
        for node in nodes:
            if node.can_fit(container):
                node.add_container(container)
                placement[container.id - 1] = node.id
                break
    
    return placement


def best_fit_placement(containers, nodes):
    """
    Метод Best Fit (z = 2)
    Контейнер размещается на узле с минимальным остатком ресурсов
    """
    placement = [0] * len(containers)

    sorted_containers = sorted(containers, key=lambda c: c.cpu + c.ram, reverse=True)
    
    for container in sorted_containers:
        best_node = None
        best_remaining = float('inf')
        
        for node in nodes:
            if node.can_fit(container):
                # Остаток ресурсов после размещения
                remaining_cpu = node.max_cpu - (node.used_cpu + container.cpu)
                remaining_ram = node.max_ram - (node.used_ram + container.ram)
                remaining = remaining_cpu + remaining_ram
                
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_node = node
        
        if best_node:
            best_node.add_container(container)
            placement[container.id - 1] = best_node.id
    
    return placement


def worst_fit_placement(containers, nodes):
    """
    Метод Worst Fit (z = 3)
    Контейнер размещается на узле с максимальным свободным местом
    """
    placement = [0] * len(containers)
    
    for container in containers:
        worst_node = None
        worst_remaining = -float('inf')
        
        for node in nodes:
            if node.can_fit(container):
                # Свободное место до размещения
                free_cpu = node.max_cpu - node.used_cpu
                free_ram = node.max_ram - node.used_ram
                free_total = free_cpu + free_ram
                
                if free_total > worst_remaining:
                    worst_remaining = free_total
                    worst_node = node
        
        if worst_node:
            worst_node.add_container(container)
            placement[container.id - 1] = worst_node.id
    
    return placement


def genetic_placement(containers, nodes, 
                      population_size=4, 
                      max_generations=10, 
                      crossover_rate=0.8, 
                      mutation_rate=0.1):
    """
    Генетический алгоритм (z = 4)
    Полностью соответствует алгоритму из "совершенно новый алгоритм.docx"
    """
    n_containers = len(containers)
    n_nodes = len(nodes)

    
    def is_valid(individual):
        """Проверка допустимости решения (нет перегрузок)"""
        # Сбрасываем узлы
        for node in nodes:
            node.used_cpu = 0
            node.used_ram = 0
        
        # Размещаем контейнеры согласно решению
        for cont_idx, node_id in enumerate(individual):
            if node_id == 0:  # Контейнер не размещен
                continue
            
            node = nodes[node_id - 1]
            container = containers[cont_idx]
            
            # Проверяем перегрузку
            if node.used_cpu + container.cpu > node.max_cpu or \
               node.used_ram + container.ram > node.max_ram:
                return False
            
            # Размещаем
            node.used_cpu += container.cpu
            node.used_ram += container.ram
        
        return True
    
    def count_placed(individual):
        """Количество размещенных контейнеров"""
        return sum(1 for x in individual if x > 0)
    
    def calculate_uniformity(individual):
        """Вычисление равномерности загрузки """
        if not is_valid(individual):
            return float('inf')
        
        # Сбрасываем и размещаем
        for node in nodes:
            node.used_cpu = 0
            node.used_ram = 0
        
        for cont_idx, node_id in enumerate(individual):
            if node_id == 0:
                continue
            node = nodes[node_id - 1]
            container = containers[cont_idx]
            node.used_cpu += container.cpu
            node.used_ram += container.ram
        
        # Собираем загрузки
        loads = []
        for node in nodes:
            if node.used_cpu > 0:  # только используемые узлы
                cpu_load = (node.used_cpu / node.max_cpu) * 100
                loads.append(cpu_load)
        
        if len(loads) <= 1:
            return 0  # один узел - идеальная равномерность
        
        # Вычисляем дисперсию
        mean_load = sum(loads) / len(loads)
        variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
        return variance
    
    def select_parents(population):
        
        # Разделяем на допустимые и недопустимые
        valid_individuals = [ind for ind in population if is_valid(ind)]
        invalid_individuals = [ind for ind in population if not is_valid(ind)]

        valid_individuals.sort(key=lambda ind: (
            -count_placed(ind),  # больше контейнеров - лучше
            calculate_uniformity(ind)  # меньше дисперсия - лучше
        ))
        
        # Сортируем недопустимые по количеству размещенных (для возможного использования)
        invalid_individuals.sort(key=lambda ind: -count_placed(ind))
        
        # Формируем список родителей
        parents = []
        
        # Берем лучших из допустимых
        parents.extend(valid_individuals[:2])
        
        # Если не хватает, добираем из недопустимых
        if len(parents) < 2 and invalid_individuals:
            parents.append(invalid_individuals[0])
        
        # Если все еще не хватает, дублируем лучшего
        while len(parents) < 2:
            parents.append(valid_individuals[0] if valid_individuals else population[0])
        
        return parents[:2]  # возвращаем ровно двух родителей
    
    
    population = []
    
    # Создаем начальную популяцию (как в документе - случайно)
    for _ in range(population_size):
        individual = []
        for _ in range(n_containers):
            # С вероятностью 0.3 контейнер не размещен (для разнообразия)
            if random.random() < 0.3:
                individual.append(0)
            else:
                individual.append(random.randint(1, n_nodes))
        population.append(individual)

    
    for generation in range(max_generations):
        # Отбираем родителей
        parent1, parent2 = select_parents(population)
        
        # Скрещивание (одноточечное)
        if random.random() < crossover_rate:
            point = random.randint(1, n_containers - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        # Мутация потомков
        for child in [child1, child2]:
            for j in range(n_containers):
                if random.random() < mutation_rate:
                    if random.random() < 0.3:
                        child[j] = 0  # сделать неразмещенным
                    else:
                        child[j] = random.randint(1, n_nodes)
        
        # Создание новой популяции (элитарность - сохраняем лучшего)
        new_population = []
        
        # Сохраняем лучшую особь из предыдущей популяции (элитарность)
        valid_inds = [ind for ind in population if is_valid(ind)]
        if valid_inds:
            best_old = min(valid_inds, key=lambda ind: calculate_uniformity(ind))
            new_population.append(best_old.copy())
        
        # Добавляем потомков
        new_population.append(child1)
        new_population.append(child2)
        
        # Заполняем оставшиеся места мутацией лучших
        while len(new_population) < population_size:
            if valid_inds:
                # Мутируем лучшего из допустимых
                best = random.choice(valid_inds[:2])
                mutant = best.copy()
                # Одна мутация
                pos = random.randint(0, n_containers - 1)
                mutant[pos] = random.randint(0, n_nodes)
                new_population.append(mutant)
            else:
                # Если нет допустимых, создаем случайную
                individual = []
                for _ in range(n_containers):
                    if random.random() < 0.3:
                        individual.append(0)
                    else:
                        individual.append(random.randint(1, n_nodes))
                new_population.append(individual)
        
        population = new_population[:population_size]
    
 
    
    # Сортируем все решения по правилам из документа
    valid_individuals = [ind for ind in population if is_valid(ind)]
    
    if valid_individuals:
        # Сортируем допустимые: по количеству контейнеров (убывание), потом по равномерности
        valid_individuals.sort(key=lambda ind: (
            -count_placed(ind),
            calculate_uniformity(ind)
        ))
        best_individual = valid_individuals[0]
    else:
        # Если нет допустимых, берем с наибольшим числом контейнеров
        best_individual = max(population, key=lambda ind: count_placed(ind))
    
    # Применяем лучшее решение к узлам
    for node in nodes:
        node.used_cpu = 0
        node.used_ram = 0
    
    for cont_idx, node_id in enumerate(best_individual):
        if node_id > 0:
            nodes[node_id - 1].add_container(containers[cont_idx])
    
    return best_individual


def calculate_metrics(placement, nodes, containers):
    """
    Вычисление всех показателей эффективности
    F₄ = CalculateMetrics(R, N)
    """
    n_nodes = len(nodes)
    n_containers = len(containers)
    
    # Количество задействованных узлов
    used_nodes = len(set(placement)) - (1 if 0 in placement else 0)
    
    # Загрузка CPU и RAM каждого узла (в процентах)
    cpu_loads = [node.get_load_cpu() for node in nodes]
    ram_loads = [node.get_load_ram() for node in nodes]
    
    # Средние значения загрузки
    avg_cpu = sum(cpu_loads) / n_nodes
    avg_ram = sum(ram_loads) / n_nodes
    
    # Процент неразмещенных контейнеров
    unplaced = sum(1 for x in placement if x == 0)
    p_unplaced = (unplaced / n_containers) * 100
    
    # Общая дисперсия загрузки узлов
    variance = 0
    for i in range(n_nodes):
        variance += (cpu_loads[i] - avg_cpu) ** 2 + (ram_loads[i] - avg_ram) ** 2
    variance /= n_nodes
    
    return {
        'placement': placement,
        'used_nodes': used_nodes,
        'cpu_loads': cpu_loads,
        'ram_loads': ram_loads,
        'avg_cpu': avg_cpu,
        'avg_ram': avg_ram,
        'p_unplaced': p_unplaced,
        'variance': variance
    }


def visualize_placement(placement, nodes, containers, method_name, filename):
    """
    Визуализация размещения для одного метода
    Соответствует рисункам из "игрушечная задача.docx"
    """
    n_nodes = len(nodes)
    
    # Подготовка данных для графика
    cpu_data = []
    ram_data = []
    node_labels = []
    
    for i, node in enumerate(nodes):
        cpu_data.append(node.get_load_cpu())
        ram_data.append(node.get_load_ram())
        node_labels.append(f"Узел {node.id}\n({node.used_cpu}/{node.max_cpu} CPU, {node.used_ram//1024}ГБ RAM)")
    
    # Создание графика
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # График 1: Размещение контейнеров
    y_pos = np.arange(n_nodes)
    ax1.barh(y_pos, cpu_data, color='skyblue', label='CPU')
    ax1.barh(y_pos, ram_data, left=cpu_data, color='lightcoral', label='RAM')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(node_labels)
    ax1.set_xlabel('Загрузка, %')
    ax1.set_title(f'{method_name}: размещение контейнеров')
    ax1.legend()
    ax1.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Максимум')
    
    # Добавляем информацию о контейнерах
    cont_text = "Размещение:\n"
    for i, node_id in enumerate(placement):
        if node_id > 0:
            cont_text += f"c{i+1} → узел {node_id}\n"
        else:
            cont_text += f"c{i+1} → НЕ РАЗМЕЩЕН\n"
    
    ax1.text(1.05, 0.5, cont_text, transform=ax1.transAxes, 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # График 2: Утилизация ресурсов
    x_pos = np.arange(n_nodes)
    width = 0.35
    
    ax2.bar(x_pos - width/2, cpu_data, width, label='CPU', color='skyblue')
    ax2.bar(x_pos + width/2, ram_data, width, label='RAM', color='lightcoral')
    ax2.set_xlabel('Узлы')
    ax2.set_ylabel('Загрузка, %')
    ax2.set_title('Утилизация ресурсов по узлам')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Узел {i+1}' for i in range(n_nodes)])
    ax2.legend()
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 120)
    
    # Добавляем значения на столбцы
    for i, v in enumerate(cpu_data):
        ax2.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(ram_data):
        ax2.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен: {filename}")


def visualize_comparison(all_metrics):
    """
    Визуализация сравнения всех методов
    """
    methods = list(all_metrics.keys())
    
    # Данные для сравнения
    used_nodes = [all_metrics[m]['used_nodes'] for m in methods]
    avg_cpu = [all_metrics[m]['avg_cpu'] for m in methods]
    avg_ram = [all_metrics[m]['avg_ram'] for m in methods]
    p_unplaced = [all_metrics[m]['p_unplaced'] for m in methods]
    variance = [all_metrics[m]['variance'] for m in methods]
    
    # Создание подграфиков
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Сравнение методов размещения контейнеров', fontsize=16, fontweight='bold')
    
    # График 1: Использованные узлы
    axes[0, 0].bar(methods, used_nodes, color=['blue', 'green', 'orange', 'red'])
    axes[0, 0].set_ylabel('Количество узлов')
    axes[0, 0].set_title('Использовано узлов')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # График 2: Средняя загрузка CPU
    axes[0, 1].bar(methods, avg_cpu, color=['blue', 'green', 'orange', 'red'])
    axes[0, 1].set_ylabel('Загрузка, %')
    axes[0, 1].set_title('Средняя загрузка CPU')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # График 3: Средняя загрузка RAM
    axes[0, 2].bar(methods, avg_ram, color=['blue', 'green', 'orange', 'red'])
    axes[0, 2].set_ylabel('Загрузка, %')
    axes[0, 2].set_title('Средняя загрузка RAM')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # График 4: Неразмещенные контейнеры
    axes[1, 0].bar(methods, p_unplaced, color=['blue', 'green', 'orange', 'red'])
    axes[1, 0].set_ylabel('Процент, %')
    axes[1, 0].set_title('Неразмещенные контейнеры')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # График 5: Дисперсия
    axes[1, 1].bar(methods, variance, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_ylabel('Дисперсия')
    axes[1, 1].set_title('Дисперсия загрузки (чем меньше, тем равномернее)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # График 6: Сравнительная таблица
    axes[1, 2].axis('off')
    
    table_data = [
        ['Метод', 'Узлов', 'CPU%', 'RAM%', 'Неразм.%', 'Дисп.'],
        ['First Fit', f"{used_nodes[0]}", f"{avg_cpu[0]:.1f}", f"{avg_ram[0]:.1f}", 
         f"{p_unplaced[0]:.1f}", f"{variance[0]:.1f}"],
        ['Best Fit', f"{used_nodes[1]}", f"{avg_cpu[1]:.1f}", f"{avg_ram[1]:.1f}", 
         f"{p_unplaced[1]:.1f}", f"{variance[1]:.1f}"],
        ['Worst Fit', f"{used_nodes[2]}", f"{avg_cpu[2]:.1f}", f"{avg_ram[2]:.1f}", 
         f"{p_unplaced[2]:.1f}", f"{variance[2]:.1f}"],
        ['Genetic', f"{used_nodes[3]}", f"{avg_cpu[3]:.1f}", f"{avg_ram[3]:.1f}", 
         f"{p_unplaced[3]:.1f}", f"{variance[3]:.1f}"]
    ]
    
    table = axes[1, 2].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Выделяем лучшие результаты
    for i, (cpu, ram, unplaced) in enumerate(zip(avg_cpu, avg_ram, p_unplaced)):
        if cpu == max(avg_cpu):
            table[(i+1, 2)].set_facecolor('lightgreen')
        if ram == max(avg_ram):
            table[(i+1, 3)].set_facecolor('lightgreen')
        if unplaced == min(p_unplaced):
            table[(i+1, 4)].set_facecolor('lightgreen')
    
    plt.tight_layout()
    plt.savefig('results/comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("График сравнения сохранен: results/comparison_table.png")


def print_metrics_table(all_metrics):
    """
    Вывод таблицы с метриками в консоль
    """
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ РАЗМЕЩЕНИЯ")
    print("="*80)
    
    table_data = []
    for method, metrics in all_metrics.items():
        table_data.append([
            method,
            metrics['used_nodes'],
            f"{metrics['avg_cpu']:.1f}%",
            f"{metrics['avg_ram']:.1f}%",
            f"{metrics['p_unplaced']:.1f}%",
            f"{metrics['variance']:.1f}"
        ])
    
    headers = ['Метод', 'Узлов', 'Ср.CPU%', 'Ср.RAM%', 'Неразм.%', 'Дисперсия']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\n" + "="*80)
    print("ДЕТАЛЬНАЯ ЗАГРУЗКА ПО УЗЛАМ")
    print("="*80)
    
    for method, metrics in all_metrics.items():
        print(f"\n{method}:")
        for i, (cpu_load, ram_load) in enumerate(zip(metrics['cpu_loads'], metrics['ram_loads'])):
            print(f"  Узел {i+1}: CPU={cpu_load:.1f}%, RAM={ram_load:.1f}%")



def run_simulator():
    """
    Основная функция симулятора
    Последовательно запускает все 4 метода и сравнивает результаты
    """
    print("="*80)
    print("СИМУЛЯТОР РАЗМЕЩЕНИЯ КОНТЕЙНЕРОВ В ВЫЧИСЛИТЕЛЬНОМ КЛАСТЕРЕ")
    print("="*80)
    
    # Входные данные 
    nodes_data = [
        (8, 16384),   # n₁: 8 ядер, 16384 МБ (16 ГБ)
        (4, 8192),    # n₂: 4 ядра, 8192 МБ (8 ГБ)
        (2, 4096)     # n₃: 2 ядра, 4096 МБ (4 ГБ)
    ]
    
    containers_data = [
        (4, 4096),    # c₁: 4 ядра, 4096 МБ (4 ГБ)
        (2, 2048),    # c₂: 2 ядра, 2048 МБ (2 ГБ)
        (2, 4096),    # c₃: 2 ядра, 4096 МБ (4 ГБ)
        (1, 1024),    # c₄: 1 ядро, 1024 МБ (1 ГБ)
        (1, 1024),    # c₅: 1 ядро, 1024 МБ (1 ГБ)
        (4, 8192)     # c₆: 4 ядра, 8192 МБ (8 ГБ)
    ]
    
    # Шаг 1-2: Инициализация
    print("\n1. Инициализация кластера и контейнеров")
    nodes = initialize_cluster(nodes_data)
    containers = initialize_containers(containers_data)
    
    print("\nУзлы кластера:")
    for node in nodes:
        print(f"  {node}")
    
    print("\nКонтейнеры:")
    for container in containers:
        print(f"  {container}")
    
    # Шаг 3: Размещение всеми методами
    print("\n" + "="*80)
    print("2. ЗАПУСК МЕТОДОВ РАЗМЕЩЕНИЯ")
    print("="*80)
    
    methods = {
        'First Fit': first_fit_placement,
        'Best Fit': best_fit_placement,
        'Worst Fit': worst_fit_placement,
        'Genetic Algorithm': lambda c, n: genetic_placement(c, n, 
                                                            population_size=4,
                                                            max_generations=10,
                                                            crossover_rate=0.8,
                                                            mutation_rate=0.1)
    }
    
    all_metrics = {}
    
    for method_name, method_func in methods.items():
        print(f"\n--- {method_name} ---")
        
        # Сброс системы
        reset_system(nodes, containers)
        
        # Запуск метода
        placement = method_func(containers, nodes)
        
        # Вывод результатов размещения
        print(f"Вектор размещения R = {placement}")
        print(f"Размещено контейнеров: {sum(1 for x in placement if x > 0)}/{len(containers)}")
        
        # Вывод загрузки узлов
        for node in nodes:
            if node.used_cpu > 0:
                print(f"  {node}")
        
        # Вычисление метрик
        metrics = calculate_metrics(placement, nodes, containers)
        all_metrics[method_name] = metrics
        
        # Визуализация для данного метода
        filename = f"results/placement_{method_name.lower().replace(' ', '_')}.png"
        visualize_placement(placement, nodes, containers, method_name, filename)
    
    # Шаг 4-5: Сравнение и визуализация
    print("\n" + "="*80)
    print("3. СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ")
    print("="*80)
    
    # Вывод таблицы с метриками
    print_metrics_table(all_metrics)
    
    # Визуализация сравнения
    visualize_comparison(all_metrics)
    


  

if __name__ == "__main__":
    # Создание папки для результатов
    import os
    os.makedirs('results', exist_ok=True)
    
    # Запуск симулятора
    run_simulator()