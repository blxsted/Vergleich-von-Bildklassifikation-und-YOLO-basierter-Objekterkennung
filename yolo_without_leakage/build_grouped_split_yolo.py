"""
build_grouped_split_yolo.py

Konvertiert PKLot-Datensatz von COCO-Format zu YOLO-Format und teilt ihn 
nach Aufnahmetag auf (GroupShuffleSplit), um Data Leakage zu vermeiden.

Workflow:
1. COCO -> YOLO Konvertierung (.txt-Labels erstellen)
2. Datum aus Dateinamen extrahieren (YYYY-MM-DD)
3. GroupShuffleSplit nach Tagen (70/15/15 train/val/test)
4. Neue Ordnerstruktur mit dataset_grouped_yolo/ erstellen
5. Validierung: Kein Tagesüberlap, alle Labels vorhanden
6. data.yaml für Colab generieren
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


class PKLotYOLOConverter:
    """Konvertiert PKLot COCO-Datensatz zu YOLO-Format mit GroupShuffleSplit."""

    def __init__(self, input_dataset_path, output_dataset_path, random_state=42):
        """
        Args:
            input_dataset_path: Pfad zum Original PKLot-Datensatz (mit train/valid/test)
            output_dataset_path: Pfad zu neuem dataset_grouped_yolo/ Ordner
            random_state: Seed für GroupShuffleSplit
        """
        self.input_path = Path(input_dataset_path)
        self.output_path = Path(output_dataset_path)
        self.random_state = random_state
        
        # COCO Klassen (aus Datensatz extrahiert)
        self.class_mapping = {
            0: "spaces",
            1: "space-empty",
            2: "space-occupied"
        }
        
        # Datenstrukturen für Tracking
        self.image_groups = defaultdict(list)  # date -> [image_paths]
        self.image_to_annotations = {}  # image_path -> [annotations]
        self.all_images = []  # alle Bilder mit Metadaten
        
    def extract_date_from_filename(self, filename):
        """
        Extrahiert Datum im Format YYYY-MM-DD aus Dateinamen.
        
        Beispiel:
            '2012-09-11_15_16_58_jpg.rf.61d961a86c9a16694403dfcb72cd450c.jpg'
            -> '2012-09-11'
        """
        match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return match.group(1)
        return None
    
    def coco_bbox_to_yolo(self, bbox, image_width, image_height):
        """
        Konvertiert COCO BBox-Format zu YOLO-Format.
        
        COCO: [x, y, width, height] (top-left corner)
        YOLO: <x_center_norm> <y_center_norm> <width_norm> <height_norm> (normalized 0-1)
        """
        x, y, w, h = bbox
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        w_norm = w / image_width
        h_norm = h / image_height
        
        # Clamp to [0, 1] (sicherheitshalber)
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        return x_center, y_center, w_norm, h_norm
    
    def load_coco_annotations(self):
        """Laden und Verarbeitung aller COCO-Annotationen aus train/valid/test."""
        print("\n" + "="*70)
        print("PHASE 1: COCO-Annotationen laden")
        print("="*70)
        
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            split_path = self.input_path / split
            coco_file = split_path / '_annotations.coco.json'
            
            if not coco_file.exists():
                print(f"⚠️  {split}: {coco_file} nicht gefunden, übersprungen")
                continue
            
            print(f"\n📂 Verarbeite {split}/ ...")
            
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Erstelle Mappings
            image_id_to_info = {img['id']: img for img in coco_data['images']}
            category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
            
            # Gruppiere Annotationen nach Image-ID
            annotations_by_image = defaultdict(list)
            for ann in coco_data['annotations']:
                annotations_by_image[ann['image_id']].append(ann)
            
            # Verarbeite jedes Bild
            for image_info in image_id_to_info.values():
                image_filename = image_info['file_name']
                image_path = split_path / image_filename
                
                # Prüfe ob Bild existiert
                if not image_path.exists():
                    continue
                
                # Extrahiere Datum
                date = self.extract_date_from_filename(image_filename)
                if not date:
                    print(f"⚠️  Konnte kein Datum extrahieren aus: {image_filename}")
                    continue
                
                # Speichere Metadaten
                image_data = {
                    'original_path': image_path,
                    'filename': image_filename,
                    'date': date,
                    'split': split,
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'image_id': image_info['id'],
                    'annotations': annotations_by_image.get(image_info['id'], [])
                }
                
                self.all_images.append(image_data)
                self.image_groups[date].append(image_data)
        
        print(f"\n✅ Geladen: {len(self.all_images)} Bilder aus {len(self.image_groups)} verschiedenen Tagen")
        return self.all_images
    
    def perform_group_shuffle_split(self, test_size=0.15, val_size=0.15):
        """
        Führt GroupShuffleSplit durch: Gruppiert nach Aufnahmetag.
        
        Das garantiert: Alle Bilder desselben Tages landen im selben Split!
        """
        print("\n" + "="*70)
        print("PHASE 2: GroupShuffleSplit nach Aufnahmetag")
        print("="*70)
        
        # Vorbereitung für GroupShuffleSplit
        dates = np.array([img['date'] for img in self.all_images])
        unique_dates = np.unique(dates)
        
        # Mapping: Datum -> numerische Gruppe
        date_to_group = {date: i for i, date in enumerate(unique_dates)}
        groups = np.array([date_to_group[date] for date in dates])
        
        indices = np.arange(len(self.all_images))
        
        print(f"\n📊 Datensatz-Statistik:")
        print(f"   - Bilder gesamt: {len(self.all_images)}")
        print(f"   - Eindeutige Tage: {len(unique_dates)}")
        print(f"   - Zeitspanne: {min(unique_dates)} bis {max(unique_dates)}")
        
        # Step 1: Trenne Test (15%) ab
        splitter_train_val = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=self.random_state
        )
        train_val_idx, test_idx = next(splitter_train_val.split(indices, groups=groups))
        
        # Step 2: Trenne Val (15% von Rest) ab
        groups_train_val = groups[train_val_idx]
        splitter_train_val2 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size / (1 - test_size),  # Adjust für den Rest
            random_state=self.random_state + 1
        )
        train_idx, val_idx = next(splitter_train_val2.split(train_val_idx, groups=groups_train_val))
        
        # Konvertiere zurück zu Original-Indizes
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]
        
        # Erstelle Splits
        splits_data = {
            'train': [self.all_images[i] for i in train_idx],
            'val': [self.all_images[i] for i in val_idx],
            'test': [self.all_images[i] for i in test_idx]
        }
        
        # Statistiken
        for split_name, split_images in splits_data.items():
            split_dates = set(img['date'] for img in split_images)
            print(f"\n{split_name}:")
            print(f"   Bilder: {len(split_images)}")
            print(f"   Eindeutige Tage: {len(split_dates)}")
            print(f"   Tage-Range: {min(split_dates)} bis {max(split_dates)}")
        
        return splits_data
    
    def validate_no_date_leakage(self, splits_data):
        """Validiert dass es 0% Tagesüberlap gibt."""
        print("\n" + "="*70)
        print("VALIDIERUNG: Tages-Überlap-Check")
        print("="*70)
        
        dates_by_split = {}
        for split_name, split_images in splits_data.items():
            dates_by_split[split_name] = set(img['date'] for img in split_images)
        
        # Prüfe auf Überlap
        train_dates = dates_by_split['train']
        val_dates = dates_by_split['val']
        test_dates = dates_by_split['test']
        
        train_val_overlap = train_dates & val_dates
        train_test_overlap = train_dates & test_dates
        val_test_overlap = val_dates & test_dates
        
        print(f"\nTrain-Val Überlap: {len(train_val_overlap)} Tage")
        print(f"Train-Test Überlap: {len(train_test_overlap)} Tage")
        print(f"Val-Test Überlap: {len(val_test_overlap)} Tage")
        
        assert len(train_val_overlap) == 0, f"❌ Train-Val Überlap: {train_val_overlap}"
        assert len(train_test_overlap) == 0, f"❌ Train-Test Überlap: {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"❌ Val-Test Überlap: {val_test_overlap}"
        
        print("\n✅ Bestätigung: 0% Tages-Überlap zwischen allen Splits!")
        
        return True
    
    def create_yolo_label_files(self, splits_data):
        """Erstellt YOLO .txt-Label-Dateien in den Output-Ordnern."""
        print("\n" + "="*70)
        print("PHASE 3: YOLO-Label-Dateien erstellen & in Output kopieren")
        print("="*70)
        
        # Erstelle Ordner-Struktur
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Verarbeite jeden Split
        split_stats = {}
        
        for split_name, split_images in splits_data.items():
            print(f"\n📁 Verarbeite {split_name}/ ({len(split_images)} Bilder)...")
            
            images_count = 0
            labels_count = 0
            
            for img_data in tqdm(split_images, desc=f"  {split_name}"):
                # Kopiere Bild
                src_image = img_data['original_path']
                dst_image = self.output_path / split_name / 'images' / img_data['filename']
                shutil.copy2(src_image, dst_image)
                images_count += 1
                
                # Erstelle YOLO-Label-Datei
                label_filename = Path(img_data['filename']).stem + '.txt'
                label_path = self.output_path / split_name / 'labels' / label_filename
                
                yolo_lines = []
                for annotation in img_data['annotations']:
                    class_id = annotation['category_id']
                    bbox = annotation['bbox']
                    
                    # Konvertiere zu YOLO-Format
                    x_center, y_center, w_norm, h_norm = self.coco_bbox_to_yolo(
                        bbox,
                        img_data['width'],
                        img_data['height']
                    )
                    
                    # Schreibe YOLO-Zeile
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    yolo_lines.append(yolo_line)
                
                # Speichere .txt-Datei
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                labels_count += 1
            
            split_stats[split_name] = {
                'images': images_count,
                'labels': labels_count
            }
        
        print(f"\n✅ YOLO-Labels erstellt und kopiert")
        return split_stats
    
    def validate_images_labels_match(self):
        """Validiert dass jedes Bild genau ein Label hat."""
        print("\n" + "="*70)
        print("VALIDIERUNG: Bilder-Labels Matching")
        print("="*70)
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_path / split / 'images'
            labels_dir = self.output_path / split / 'labels'
            
            image_files = set(f.stem for f in images_dir.glob('*.jpg'))
            label_files = set(f.stem for f in labels_dir.glob('*.txt'))
            
            missing_labels = image_files - label_files
            extra_labels = label_files - image_files
            
            assert len(missing_labels) == 0, f"❌ {split}: {len(missing_labels)} Bilder ohne Labels"
            assert len(extra_labels) == 0, f"❌ {split}: {len(extra_labels)} Label ohne Bilder"
            
            print(f"✅ {split}: {len(image_files)} Bilder = {len(label_files)} Labels")
        
        return True
    
    def create_data_yaml(self):
        """Erstellt data.yaml für Colab (Google Drive paths)."""
        print("\n" + "="*70)
        print("PHASE 4: data.yaml für Google Colab erstellen")
        print("="*70)
        
        # Für Colab: Google Drive Paths
        data_yaml_content = """# PKLot YOLO Dataset (GroupShuffleSplit nach Aufnahmetag)
# Konfiguriert für Google Colab mit Google Drive

path: /content/drive/MyDrive/dataset_grouped_yolo
train: train/images
val: val/images
test: test/images

nc: 3
names: ['spaces', 'space-empty', 'space-occupied']

# Metadaten
description: |
  PKLot Dataset - YOLO Format
  Konvertiert von COCO, aufgeteilt nach Aufnahmetag (GroupShuffleSplit)
  Garantiert: 0% Tages-Überlap zwischen train/val/test
  
  Klassen:
  - 0: spaces (generischer Parkplatz)
  - 1: space-empty (leerer Parkplatz)
  - 2: space-occupied (besetzter Parkplatz)
"""
        
        yaml_path = self.output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(data_yaml_content)
        
        print(f"✅ data.yaml erstellt: {yaml_path}")
        return yaml_path
    
    def print_final_summary(self, splits_data):
        """Gibt finales Summary aus."""
        print("\n" + "="*70)
        print("FINALE ZUSAMMENFASSUNG")
        print("="*70)
        
        print(f"\n📊 Datensatz-Struktur:")
        print(f"Output-Ordner: {self.output_path}")
        print(f"\nBilder und Labels pro Split:")
        
        all_dates = set()
        for split_name, split_images in splits_data.items():
            dates = set(img['date'] for img in split_images)
            all_dates.update(dates)
            
            print(f"\n  {split_name.upper()}:")
            print(f"    ├─ Bilder: {len(split_images)}")
            print(f"    ├─ Labels: {len(split_images)} (1:1 Match)")
            print(f"    └─ Eindeutige Tage: {len(dates)}")
        
        print(f"\nGesamt eindeutige Tage: {len(all_dates)}")
        print(f"\n✅ Tages-Überlap Validierung: BESTANDEN (0% Überlap)")
        print(f"✅ Bilder-Labels Matching: BESTANDEN (100% Match)")
        
        print(f"\n📁 Neue Datensatz-Struktur:")
        print(f"   dataset_grouped_yolo/")
        print(f"   ├─ train/")
        print(f"   │  ├─ images/ ({len(splits_data['train'])} Bilder)")
        print(f"   │  └─ labels/ ({len(splits_data['train'])} Labels)")
        print(f"   ├─ val/")
        print(f"   │  ├─ images/ ({len(splits_data['val'])} Bilder)")
        print(f"   │  └─ labels/ ({len(splits_data['val'])} Labels)")
        print(f"   ├─ test/")
        print(f"   │  ├─ images/ ({len(splits_data['test'])} Bilder)")
        print(f"   │  └─ labels/ ({len(splits_data['test'])} Labels)")
        print(f"   └─ data.yaml")
        
        print("\n" + "="*70)
        print("✅ DATENSATZ ERFOLGREICH KONVERTIERT!")
        print("="*70)
    
    def run(self):
        """Hauptfunktion: Alle Phasen ausführen."""
        try:
            # Phase 1: Lade COCO-Annotationen
            self.load_coco_annotations()
            
            # Phase 2: GroupShuffleSplit nach Datum
            splits_data = self.perform_group_shuffle_split()
            
            # Validierung: Keine Tages-Überlap
            self.validate_no_date_leakage(splits_data)
            
            # Phase 3: YOLO-Labels erstellen
            split_stats = self.create_yolo_label_files(splits_data)
            
            # Validierung: Bilder-Labels Matching
            self.validate_images_labels_match()
            
            # Phase 4: data.yaml erstellen
            self.create_data_yaml()
            
            # Finale Zusammenfassung
            self.print_final_summary(splits_data)
            
            print("\n🎉 Alles abgeschlossen! Ready für Google Colab!\n")
            
        except Exception as e:
            print(f"\n❌ FEHLER: {e}")
            raise


def main():
    """Entry-Point des Skripts."""
    
    # Pfade
    input_dataset = "/Users/blvsted/Dateien/01. Active/MS/Vergleich-von-Bildklassifikation-und-YOLO-basierter-Objekterkennung/dataset"
    output_dataset = "/Users/blvsted/Dateien/01. Active/MS/Vergleich-von-Bildklassifikation-und-YOLO-basierter-Objekterkennung/dataset_grouped_yolo"
    
    # Initialisiere und führe aus
    converter = PKLotYOLOConverter(input_dataset, output_dataset)
    converter.run()


if __name__ == "__main__":
    main()
