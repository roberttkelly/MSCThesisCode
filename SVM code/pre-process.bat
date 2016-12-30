SET dir=20000
SET in_dir=data\train-%dir%
SET out_dir=processed
if not exist %out_dir%\%in_dir% mkdir %out_dir%\%in_dir%

SET size=512x512

for %%i in (%in_dir%\*) do convert -fuzz 10%% -trim +repage -resize %size% -gravity center -background black -extent %size% -equalize -colorspace gray %%i %out_dir%\%%i

ren %out_dir%\%in_dir% %dir%-%size%