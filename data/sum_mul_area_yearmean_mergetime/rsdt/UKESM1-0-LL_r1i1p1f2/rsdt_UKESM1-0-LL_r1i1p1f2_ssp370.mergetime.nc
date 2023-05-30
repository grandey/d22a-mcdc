CDF   �   
      time       bnds      lon       lat          0   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �UKESM1.0-LL (2018): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: UKCA-StratTrop
land: JULES-ES-1.0
landIce: none
ocean: NEMO-HadGEM3-GO6.0 (eORCA1 tripolar primarily 1 deg with meridional refinement down to 1/3 degree in the tropics; 360 x 330 longitude/latitude; 75 levels; top grid cell 0-1 m)
ocnBgchem: MEDUSA2
seaIce: CICE-HadGEM3-GSI8 (eORCA1 tripolar primarily 1 deg; 360 x 330 longitude/latitude)   institution       BMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK     activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A�        creation_date         2019-04-05T16:01:49Z   
cv_version        6.2.20.1   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Nhttps://furtherinfo.es-doc.org/CMIP6.MOHC.UKESM1-0-LL.historical.none.r1i1p1f2     grid      -Native N96 grid; 192 x 144 longitude/latitude      
grid_label        gn     history      	�Tue May 30 16:58:31 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/UKESM1-0-LL_r1i1p1f2/rsdt_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp370.r1i1p1f2.Amon.rsdt.gn.v20190510/rsdt_Amon_UKESM1-0-LL_ssp370_r1i1p1f2_gn_201501-204912.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp370.r1i1p1f2.Amon.rsdt.gn.v20190510/rsdt_Amon_UKESM1-0-LL_ssp370_r1i1p1f2_gn_205001-210012.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/UKESM1-0-LL_r1i1p1f2/rsdt_UKESM1-0-LL_r1i1p1f2_ssp370.mergetime.nc
Tue May 30 16:58:30 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.yearmean.mul.areacella_piControl_v20190705.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/UKESM1-0-LL_r1i1p1f2/rsdt_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc
Fri Nov 04 05:52:28 2022: cdo -O -s -fldsum -setattribute,rsdt@units=W m-2 m2 -mul -yearmean -selname,rsdt /Users/benjamin/Data/p22b/CMIP6/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.piControl.r1i1p1f2.fx.areacella.gn.v20190705/areacella_fx_UKESM1-0-LL_piControl_r1i1p1f2_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.yearmean.mul.areacella_piControl_v20190705.fldsum.nc
2019-04-05T15:50:03Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
2019-04-05T15:49:43Z MIP Convert v1.0.2, Python v2.7.12, Iris v1.13.0, Numpy v1.13.3, netcdftime v1.4.1.   initialization_index            institution_id        MOHC   mip_era       CMIP6      mo_runid      u-bc179    nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      UKESM1-0-LL    parent_time_units         days since 1850-01-01-00-00-00     parent_variant_label      r1i1p1f2   physics_index               product       model-output   realization_index               realm         atmos      	source_id         UKESM1-0-LL    source_type       AOGCM AER BGC CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(13 December 2018) MD5:2b12b5db6db112aa8b8b0d6c1645b121      title         %UKESM1-0-LL output prepared for CMIP6      variable_id       rsdt   variant_label         r1i1p1f2   license      XCMIP6 model data produced by the Met Office Hadley Centre is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https://ukesm.ac.uk/cmip6. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   cmor_version      3.4.0      tracking_id       1hdl:21.14100/cc9cff0b-2c6b-4fc4-8674-8c9ab1bf81f6      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T               |   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               l   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               t   rsdt                   
   standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       9Shortwave radiation incident at the top of the atmosphere      original_name         $mo: (stash: m01s01i207, lbproc: 128)   cell_measures         area: areacella    history       u2019-04-05T16:01:49Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Aq���   Aq��P   Aq�P   \3�Aq�6�   Aq�P   Aq��P   \2oAq���   Aq��P   Aq��P   \2QAq��   Aq��P   Aq�dP   \0Aq���   Aq�dP   Aq��P   \,�Aq���   Aq��P   Aq�FP   \*�Aq�k�   Aq�FP   Aq��P   \*aAq���   Aq��P   Aq�(P   \+�Aq�M�   Aq�(P   Aq��P   \/�Aq���   Aq��P   Aq�
P   \2�Aq�/�   Aq�
P   Aq�{P   \54Aq���   Aq�{P   Aq��P   \4�Aq��   Aq��P   Aq�]P   \1�AqĂ�   Aq�]P   Aq��P   \0VAq���   Aq��P   Aq�?P   \.�Aq�d�   Aq�?P   Aq˰P   \-rAq���   Aq˰P   Aq�!P   \,�Aq�F�   Aq�!P   AqВP   \+�Aqз�   AqВP   Aq�P   \.Aq�(�   Aq�P   Aq�tP   \2	Aqՙ�   Aq�tP   Aq��P   \2\Aq�
�   Aq��P   Aq�VP   \4�Aq�{�   Aq�VP   Aq��P   \3tAq���   Aq��P   Aq�8P   \1WAq�]�   Aq�8P   Aq�P   \/nAq���   Aq�P   Aq�P   \+�Aq�?�   Aq�P   Aq�P   \*DAq��   Aq�P   Aq��P   \);Aq�!�   Aq��P   Aq�mP   \'�Aq��   Aq�mP   Aq��P   \(1Aq��   Aq��P   Aq�OP   \+_Aq�t�   Aq�OP   Aq��P   \/�Aq���   Aq��P   Aq�1P   \.�Aq�V�   Aq�1P   Aq��P   \..Aq���   Aq��P   Aq�P   \3cAq�8�   Aq�P   Aq��P   \0!Aq���   Aq��P   Aq��P   \,kAq��   Aq��P   ArfP   \*Ar��   ArfP   Ar�P   \)1Ar��   Ar�P   ArHP   \(�Arm�   ArHP   Ar�P   \)�Ar��   Ar�P   Ar*P   \/kArO�   Ar*P   Ar�P   \4)Ar��   Ar�P   ArP   \7�Ar1�   ArP   Ar}P   \<oAr��   Ar}P   Ar�P   \83Ar�   Ar�P   Ar_P   \2VAr��   Ar_P   Ar�P   \.�Ar��   Ar�P   ArAP   \,�Arf�   ArAP   Ar�P   \*�Ar��   Ar�P   Ar!#P   \)�Ar!H�   Ar!#P   Ar#�P   \'�Ar#��   Ar#�P   Ar&P   \'�Ar&*�   Ar&P   Ar(vP   \,*Ar(��   Ar(vP   Ar*�P   \3�Ar+�   Ar*�P   Ar-XP   \0QAr-}�   Ar-XP   Ar/�P   \5�Ar/��   Ar/�P   Ar2:P   \2�Ar2_�   Ar2:P   Ar4�P   \4�Ar4��   Ar4�P   Ar7P   \0�Ar7A�   Ar7P   Ar9�P   \.Ar9��   Ar9�P   Ar;�P   \* Ar<#�   Ar;�P   Ar>oP   \(mAr>��   Ar>oP   Ar@�P   \(�ArA�   Ar@�P   ArCQP   \+�ArCv�   ArCQP   ArE�P   \4�ArE��   ArE�P   ArH3P   \<0ArHX�   ArH3P   ArJ�P   \@�ArJ��   ArJ�P   ArMP   \A�ArM:�   ArMP   ArO�P   \:bArO��   ArO�P   ArQ�P   \3eArR�   ArQ�P   ArThP   \/ArT��   ArThP   ArV�P   \+ArV��   ArV�P   ArYJP   \*+ArYo�   ArYJP   Ar[�P   \+�Ar[��   Ar[�P   Ar^,P   \0�Ar^Q�   Ar^,P   Ar`�P   \7Ar`��   Ar`�P   ArcP   \<�Arc3�   ArcP   AreP   \9Are��   AreP   Arg�P   \7Arh�   Arg�P   ArjaP   \7TArj��   ArjaP   Arl�P   \2"Arl��   Arl�P   AroCP   \.rAroh�   AroCP   Arq�P   \,�Arq��   Arq�P   Art%P   \.�ArtJ�   Art%P   Arv�P   \4�Arv��   Arv�P   AryP   \A�Ary,�   AryP   Ar{xP   \B�Ar{��   Ar{xP   Ar}�P   \@Ar~�   Ar}�P   Ar�ZP   \?�Ar��   Ar�ZP   Ar��P   \;Ar���   Ar��P   Ar�<P   \7�Ar�a�   Ar�<P   Ar��P   \3:Ar���   Ar��P   Ar�P   \.�Ar�C�   Ar�P   Ar��P   \0AAr���   Ar��P   Ar� P   \8FAr�%�   Ar� P   Ar�qP   \;�Ar���   Ar�qP   Ar��P   \FgAr��   Ar��P   Ar�SP   \JAr�x�   Ar�SP   Ar��P   \FiAr���   Ar��P   Ar�5P   \>�Ar�Z�   Ar�5P   Ar��P   \4�Ar���   Ar��P   Ar�P   \4iAr�<�   Ar�P   Ar��P   \1xAr���   Ar��P   Ar��P   \1{Ar��   Ar��P   Ar�jP   \7�Ar���   Ar�jP   Ar��P   \E1Ar� �   Ar��P   Ar�LP   \R%Ar�q�   Ar�LP   Ar��P   \S�Ar���   Ar��P   Ar�.P   \L�Ar�S�   Ar�.P   Ar��P   \G�Ar���   Ar��P   Ar�P   \<�Ar�5�   Ar�P   Ar��P   \5	Ar���   Ar��P   Ar��P   \2|Ar��   Ar��P   Ar�cP   \1DAr���   Ar�cP   Ar��P   \1�Ar���   Ar��P   Ar�EP   \7�Ar�j�   Ar�EP   ArĶP   \>�Ar���   ArĶP   Ar�'P   \B�Ar�L�   Ar�'P   ArɘP   \CeArɽ�   ArɘP   Ar�	P   \C�Ar�.�   Ar�	P   Ar�zP   \=eArΟ�   Ar�zP   Ar��P   \<�Ar��   Ar��P   Ar�\P   \6kArӁ�   Ar�\P   Ar��P   \37Ar���   Ar��P   Ar�>P   \3HAr�c�   Ar�>P   ArگP   \28Ar���   ArگP   Ar� P   \9Ar�E�   Ar� P   ArߑP   \C`Ar߶�   ArߑP   Ar�P   \LWAr�'�   Ar�P   Ar�sP   \J�Ar��   Ar�sP   Ar��P   \K4Ar�	�   Ar��P   Ar�UP   \AHAr�z�   Ar�UP   Ar��P   \@�Ar���   Ar��P   Ar�7P   \5;Ar�\�   Ar�7P   Ar�P   \1qAr���   Ar�P   Ar�P   \1�Ar�>�   Ar�P   Ar��P   \4_Ar���   Ar��P   Ar��P   \>�Ar� �   Ar��P   Ar�lP   \L�Ar���   Ar�lP   Ar��P   \JAr��   Ar��P   Ar�NP   \I�Ar�s�   Ar�NP   As�P   \EAs��   As�P   As0P   \9�AsU�   As0P   As�P   \33As��   As�P   As	P   \1�As	7�   As	P   As�P   \/As��   As�P   As�P   \2�As�   As�P   AseP   \=mAs��   AseP   As�P   \EAAs��   As�P   AsGP   \KAsl�   AsGP   As�P   \H�As��   As�P   As)P   \JAsN�   As)P   As�P   \<As��   As�P   AsP   \5�As0�   AsP   As!|P   \/�As!��   As!|P   As#�P   \.nAs$�   As#�P   As&^P   \+�As&��   As&^P   As(�P   \+CAs(��   As(�P   As+@P   \+=As+e�   As+@P   As-�P   \0GAs-��   As-�P   As0"P   \8As0G�   As0"P   As2�P   \;_As2��   As2�P   As5P   \=%As5)�   As5P   As7uP   \?bAs7��   As7uP   As9�P   \8�As:�   As9�P   As<WP   \0HAs<|�   As<WP   As>�P   \.As>��   As>�P   AsA9P   \-(AsA^�   AsA9P   AsC�P   \-AsC��   AsC�P   AsFP   \./AsF@�   AsFP   AsH�P   \6rAsH��   AsH�P   AsJ�P   \?nAsK"�   AsJ�P   AsMnP   \D�AsM��   AsMnP   AsO�P   \F�AsP�   AsO�P   AsRPP   \A�AsRu�   AsRPP   AsT�P   \;�AsT��   AsT�P   AsW2P   \6,AsWW�   AsW2P   AsY�P   \3�AsY��   AsY�P   As\P   \0As\9�   As\P   As^�P   \-�As^��   As^�P   As`�P   \+�Asa�   As`�P   AscgP   \,dAsc��   AscgP   Ase�P   \1TAse��   Ase�P   AshIP   \9�Ashn�   AshIP   Asj�P   \:�Asj��   Asj�P   Asm+P   \>�AsmP�   Asm+P   Aso�P   \:�Aso��   Aso�P   AsrP   \:�Asr2�   AsrP   Ast~P   \5XAst��   Ast~P   Asv�P   \0Asw�   Asv�P   Asy`P   \+}Asy��   Asy`P   As{�P   \*6As{��   As{�P   As~BP   \*�As~g�   As~BP   As��P   \-�As���   As��P   As�$P   \5�As�I�   As�$P   As��P   \;jAs���   As��P   As�P   \A�As�+�   As�P   As�wP   \BPAs���   As�wP   As��P   \="As��   As��P   As�YP   \5\As�~�   As�YP   As��P   \1�As���   As��P   As�;P   \-�As�`�   As�;P   As��P   \+fAs���   As��P   As�P   \,aAs�B�   As�P   As��P   \1As���   As��P   As��P   \6�As�$�   As��P   As�pP   \7+As���   As�pP   As��P   \83As��   As��P   As�RP   \8�As�w�   As�RP   As��P   \3�As���   As��P   As�4P   \/�As�Y�   As�4P   As��P   \,jAs���   As��P   As�P   \+<As�;�   As�P   As��P   \*�As���   As��P   As��P   \,DAs��   As��P   As�iP   \2�As���   As�iP   As��P   \8rAs���   As��P   As�KP   \;�As�p�   As�KP   As��P   \:iAs���   As��P   As�-P   \6�As�R�   As�-P   AsP   \1�As���   AsP   As�P   \/BAs�4�   As�P   AsǀP   \-vAsǥ�   AsǀP   As��P   \*�As��   As��P   As�bP   \)�Aṡ�   As�bP   As��P   \(�As���   As��P   As�DP   \*As�i�   As�DP   AsӵP   \. As���   AsӵP   As�&P   \2qAs�K�   As�&P   AsؗP   \5�Asؼ�   AsؗP   As�P   \5�As�-�   As�P   As�yP   \5�Asݞ�   As�yP   As��P   \2{As��   As��P   As�[P   \0�As��   As�[P   As��P   \,As���   As��P   As�=P   \)LAs�b�   As�=P   As�P   \)As���   As�P   As�P   \)sAs�D�   As�P   As�P   \-?As��   As�P   As�P   \3As�&�   As�P   As�rP   \8As��   As�rP   As��P   \>)As��   As��P   As�TP   \;�As�y�   As�TP   As��P   \6�As���   As��P   As�6P   \2�As�[�   As�6P   As��P   \-�As���   As��P   AtP   \,`At=�   AtP   At�P   \*�At��   At�P   At�P   \-At�   At�P   At	kP   \1�