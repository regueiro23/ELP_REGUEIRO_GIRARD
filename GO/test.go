package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Kagami/go-face"
)

const dataDirSamples = "testdata/samples"
const dataDirImages = "testdata/images"
const dataDirModels = "testdata/models"
const dataDirResultats = "testdata/resultats"
const dossierSortie = "resultat"

var photosBase = []string{"pierre.jpg", "mohammed.jpg", "nayoung.jpg", "chaeyeon.jpg"}
var photosComparees []string

func main() {

	startTime := time.Now()

	fmt.Println("Reconnaissance 3000")

	//////////////////////////////////////////////////////
	//
	// ON INITIALISE LE RECONNAISSEUR GRACE AUX MODELES
	//
	//////////////////////////////////////////////////////

	rec, err := face.NewRecognizer(dataDirModels)
	if err != nil {
		fmt.Println("Impossible d'initialiser le modèle de reconnaissance faciale")
	}
	defer rec.Close()
	fmt.Println("Modèle de reconnaissance initialisé")

	/////////////////////////////////////////////////////
	//
	// ON ANALYSE LES VISAGES EN ENTREE
	//
	/////////////////////////////////////////////////////

	var samples []face.Descriptor
	var visage face.Face
	var labels []string
	var identifiants []int32

	for indice, image := range photosBase {
		fmt.Println("Analyse/Sampling du visage présent sur", image)
		visage = sampleVisage(rec, image)
		samples = append(samples, visage.Descriptor)
		labels = append(labels, strings.TrimSuffix(image, ".jpg"))
		identifiants = append(identifiants, int32(indice))
	}

	rec.SetSamples(samples, identifiants)

	////////////////////////////////////////////////////////
	//
	// ON COMPARE LES VISAGES D'UNE PHOTO A NOS SAMPLES
	//
	////////////////////////////////////////////////////////

	photosComparees = recupererFichiers(dataDirImages)

	for _, image := range photosComparees {

		fmt.Println("Image :", image)
		visagesCompares := sampleMultiplesVisages(rec, image)

		if visagesCompares == nil {
			fmt.Println("Aucun visage sur cette image")
		} else {
			for _, visage := range visagesCompares {
				IDVisage := rec.ClassifyThreshold(visage.Descriptor, 0.15)
				if IDVisage < 0 {
					fmt.Println("Ne peut pas classifier")
				} else {
					fmt.Println(labels[IDVisage])
					fmt.Println(visage.Rectangle.Min)
					fmt.Println(visage.Rectangle.Max)
					enregistreCopieRectangle(image, visage.Rectangle.Min.X, visage.Rectangle.Min.Y, visage.Rectangle.Max.X, visage.Rectangle.Max.Y, labels[IDVisage], image)

				}

			}
		}

	}

	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	fmt.Printf("Temps d'exécution total : %s\n", elapsedTime)
	////////////////////////////////////////////////////////////

}

func sampleVisage(rec *face.Recognizer, photo string) face.Face {
	fichierImage := filepath.Join(dataDirSamples, photo)
	visage, err := rec.RecognizeSingleFile(fichierImage)
	if err != nil {
		log.Fatalf("Can't recognize: %v", photo)
	}
	return *visage
}

func sampleMultiplesVisages(rec *face.Recognizer, photo string) []face.Face {
	fichierImage := filepath.Join(dataDirImages, photo)
	liste_visages, err := rec.RecognizeFile(fichierImage)
	if err != nil {
		log.Fatalf("Can't recognize: %v", photo)
	}
	return liste_visages
}

func enregistreCopieRectangle(inputImageName string, x1, y1, x2, y2 int, outputDir string, outputImageName string) {
	// Ouvrir le fichier image
	inputImagePath := filepath.Join(dataDirImages, inputImageName)
	inputImageFile, err := os.Open(inputImagePath)
	if err != nil {
		log.Fatalf("Erreur ouverture fichier pour modification : %s", inputImageName)
	}
	defer inputImageFile.Close()

	// Décoder le fichier image
	img, _, err := image.Decode(inputImageFile)
	if err != nil {
		log.Fatalf("Erreur décodage fichier pour modification : %s", inputImageName)
	}

	// Créer un nouvel image RGBA pour dessiner le rectangle rouge
	bounds := img.Bounds()
	rgbaImg := image.NewRGBA(bounds)
	draw.Draw(rgbaImg, bounds, img, image.Point{}, draw.Over)

	// Dessiner les contours du rectangle rouge
	red := color.RGBA{255, 0, 0, 255} // Rouge pur, sans mélange
	draw.Draw(rgbaImg, image.Rect(x1, y1, x2, y1+1), &image.Uniform{red}, image.Point{}, draw.Over)
	draw.Draw(rgbaImg, image.Rect(x1, y1, x1+1, y2), &image.Uniform{red}, image.Point{}, draw.Over)
	draw.Draw(rgbaImg, image.Rect(x2-1, y1, x2, y2), &image.Uniform{red}, image.Point{}, draw.Over)
	draw.Draw(rgbaImg, image.Rect(x1, y2-1, x2, y2), &image.Uniform{red}, image.Point{}, draw.Over)

	// Créer le dossier de sortie :
	cheminDossier := filepath.Join(dataDirResultats, outputDir)
	_, err = os.Stat(cheminDossier)

	if os.IsNotExist(err) {
		// Le dossier n'existe pas, le créer
		err := os.MkdirAll(cheminDossier, os.ModePerm)
		if err != nil {
			log.Fatalf("Erreur création de dossier pour l'image: %s", inputImageName)
		}
		fmt.Printf("Dossier '%s' créé.\n", cheminDossier)
	} else if err != nil {
		// Une erreur s'est produite lors de la vérification
		log.Fatalf("Erreur lors de la vérification du dossier pour l'image: %s", inputImageName)
	} else {
		// Le dossier existe déjà
		fmt.Printf("Le dossier '%s' existe déjà.\n", cheminDossier)
	}

	// Créer le fichier de sortie
	outputImagePath := filepath.Join(dataDirResultats, outputDir, outputImageName)
	outputImageFile, err := os.Create(outputImagePath)
	if err != nil {
		log.Fatalf("Erreur lors de la création du fichier de sortie pour l'image: %s", inputImageName)
	}
	defer outputImageFile.Close()

	// Encoder l'image résultante au format JPEG
	err = jpeg.Encode(outputImageFile, rgbaImg, nil)
	if err != nil {
		log.Fatalf("Erreur d'encodage du fichier pour l'image: %s", inputImageName)
	}
}

func recupererFichiers(dossierSource string) []string {
	var listeFichiers []string

	// Lire le contenu du dossier
	contenuDossier, err := ioutil.ReadDir(dossierSource)
	if err != nil {
		log.Fatalf("Soucis collecte fichiers")
	}

	// Parcourir les fichiers du dossier
	for _, fichier := range contenuDossier {
		// Vérifier si le fichier a l'extension .jpg
		if fichier.IsDir() == false && filepath.Ext(fichier.Name()) == ".jpg" {
			listeFichiers = append(listeFichiers, fichier.Name())
		}
	}

	return listeFichiers
}
