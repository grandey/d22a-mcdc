CDF   �   
      time       bnds      lon       lat          7   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       ,CanESM5 (2019): 
aerosol: interactive
atmos: CanAM5 (T63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude; 49 levels; top level 1 hPa)
atmosChem: specified oxidants for aerosols
land: CLASS3.6/CTEM1.2
landIce: specified ice sheets
ocean: NEMO3.4.1 (ORCA1 tripolar grid, 1 deg with refinement to 1/3 deg within 20 degrees of the equator; 361 x 290 longitude/latitude; 45 vertical levels; top grid cell 0-6.19 m)
ocnBgchem: Canadian Model of Ocean Carbon (CMOC); NPZD ecosystem with OMIP prescribed carbonate chemistry
seaIce: LIM2   institution       wCanadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada, Victoria, BC V8P 5C2, Canada    CCCma_model_hash      (3dedf95315d603326fde4f5340dc0519d80d10c0   CCCma_parent_runid        
rc3-pictrl     CCCma_pycmor_hash         (33c30511acc319a98240633965a04ca99c26427e   CCCma_runid       rc3.1-his01    YMDH_branch_time_in_child         1850:01:01:00      YMDH_branch_time_in_parent        5201:01:01:00      activity_id       CMIP   branch_method         Spin-up documentation      branch_time_in_child                 branch_time_in_parent         A2��       contact       %ec.cccma.info-info.ccmac.ec@canada.ca      creation_date         2019-04-30T17:32:13Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Khttps://furtherinfo.es-doc.org/CMIP6.CCCma.CanESM5.historical.none.r1i1p1f1    grid      kT63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude; 49 levels; top level 1 hPa    
grid_label        gn     history      DWed Nov 09 18:59:18 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/CanESM5_r1i1p1f1/rsdt_CanESM5_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CanESM5_r1i1p1f1/CMIP6.ScenarioMIP.CCCma.CanESM5.ssp370.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_ssp370_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp370_v20190429.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/CanESM5_r1i1p1f1/rsdt_CanESM5_r1i1p1f1_ssp370.mergetime.nc
Wed Nov 09 18:59:16 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190429.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/CanESM5_r1i1p1f1/rsdt_CanESM5_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 04:42:49 2022: cdo -O -s -fldsum -setattribute,rsdt@units=W m-2 m2 -mul -yearmean -selname,rsdt /Users/benjamin/Data/p22b/CMIP6/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.fx.areacella.gn.v20190429/areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190429.fldsum.nc
2019-04-30T17:32:13Z ;rewrote data to be consistent with CMIP for variable rsdt found in table Amon.;
Output from $runid   initialization_index            institution_id        CCCma      mip_era       CMIP6      nominal_resolution        500 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      CanESM5    parent_time_units         days since 1850-01-01 0:0:0.0      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      
references        mGeophysical Model Development Special issue on CanESM5 (https://www.geosci-model-dev.net/special_issues.html)      	source_id         CanESM5    source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(20 February 2019) MD5:374fbe5a2bcca535c40f7f23da271e49      title         !CanESM5 output prepared for CMIP6      tracking_id       1hdl:21.14100/1ab756cd-1928-40f9-82dc-a0e58a183ed8      variable_id       rsdt   variant_label         r1i1p1f1   version       	v20190429      license      �CMIP6 model data produced by The Government of Canada (Canadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada) is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsdt                   
   standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       9Shortwave radiation incident at the top of the atmosphere      original_name         FSO    cell_measures         area: areacella    history       �2019-04-30T17:32:13Z altered by CMOR: Reordered dimensions, original order: lat lon time. 2019-04-30T17:32:13Z altered by CMOR: replaced missing value flag (1e+38) with standard missing value (1e+20).                            Aq���   Aq��P   Aq�P   \)?Aq�6�   Aq�P   Aq��P   \'FAq���   Aq��P   Aq��P   \'Aq��   Aq��P   Aq�dP   \%KAq���   Aq�dP   Aq��P   \"Aq���   Aq��P   Aq�FP   \ 3Aq�k�   Aq�FP   Aq��P   \yAq���   Aq��P   Aq�(P   \ �Aq�M�   Aq�(P   Aq��P   \$Aq���   Aq��P   Aq�
P   \'@Aq�/�   Aq�
P   Aq�{P   \*Aq���   Aq�{P   Aq��P   \)�Aq��   Aq��P   Aq�]P   \'AqĂ�   Aq�]P   Aq��P   \%sAq���   Aq��P   Aq�?P   \#�Aq�d�   Aq�?P   Aq˰P   \"�Aq���   Aq˰P   Aq�!P   \!�Aq�F�   Aq�!P   AqВP   \!Aqз�   AqВP   Aq�P   \"�Aq�(�   Aq�P   Aq�tP   \&�Aqՙ�   Aq�tP   Aq��P   \&�Aq�
�   Aq��P   Aq�VP   \)}Aq�{�   Aq�VP   Aq��P   \(Aq���   Aq��P   Aq�8P   \&�Aq�]�   Aq�8P   Aq�P   \$�Aq���   Aq�P   Aq�P   \!4Aq�?�   Aq�P   Aq�P   \�Aq��   Aq�P   Aq��P   \bAq�!�   Aq��P   Aq�mP   \DAq��   Aq�mP   Aq��P   \5Aq��   Aq��P   Aq�OP   \�Aq�t�   Aq�OP   Aq��P   \$iAq���   Aq��P   Aq�1P   \#^Aq�V�   Aq�1P   Aq��P   \#XAq���   Aq��P   Aq�P   \(ZAq�8�   Aq�P   Aq��P   \%
Aq���   Aq��P   Aq��P   \"!Aq��   Aq��P   ArfP   \�Ar��   ArfP   Ar�P   \Ar��   Ar�P   ArHP   \�Arm�   ArHP   Ar�P   \�Ar��   Ar�P   Ar*P   \#�ArO�   Ar*P   Ar�P   \(YAr��   Ar�P   ArP   \-Ar1�   ArP   Ar}P   \0`Ar��   Ar}P   Ar�P   \-OAr�   Ar�P   Ar_P   \'�Ar��   Ar_P   Ar�P   \$	Ar��   Ar�P   ArAP   \!�Arf�   ArAP   Ar�P   \ EAr��   Ar�P   Ar!#P   \�Ar!H�   Ar!#P   Ar#�P   \�Ar#��   Ar#�P   Ar&P   \Ar&*�   Ar&P   Ar(vP   \ Ar(��   Ar(vP   Ar*�P   \(RAr+�   Ar*�P   Ar-XP   \$�Ar-}�   Ar-XP   Ar/�P   \+bAr/��   Ar/�P   Ar2:P   \'qAr2_�   Ar2:P   Ar4�P   \)gAr4��   Ar4�P   Ar7P   \%�Ar7A�   Ar7P   Ar9�P   \#�Ar9��   Ar9�P   Ar;�P   \rAr<#�   Ar;�P   Ar>oP   \�Ar>��   Ar>oP   Ar@�P   \�ArA�   Ar@�P   ArCQP   \ kArCv�   ArCQP   ArE�P   \(�ArE��   ArE�P   ArH3P   \0�ArHX�   ArH3P   ArJ�P   \5�ArJ��   ArJ�P   ArMP   \6|ArM:�   ArMP   ArO�P   \/$ArO��   ArO�P   ArQ�P   \(�ArR�   ArQ�P   ArThP   \$�ArT��   ArThP   ArV�P   \ �ArV��   ArV�P   ArYJP   \�ArYo�   ArYJP   Ar[�P   \ �Ar[��   Ar[�P   Ar^,P   \&�Ar^Q�   Ar^,P   Ar`�P   \)�Ar`��   Ar`�P   ArcP   \0�Arc3�   ArcP   AreP   \-�Are��   AreP   Arg�P   \-MArh�   Arg�P   ArjaP   \*�Arj��   ArjaP   Arl�P   \'�Arl��   Arl�P   AroCP   \#�Aroh�   AroCP   Arq�P   \!�Arq��   Arq�P   Art%P   \#NArtJ�   Art%P   Arv�P   \)dArv��   Arv�P   AryP   \4�Ary,�   AryP   Ar{xP   \6�Ar{��   Ar{xP   Ar}�P   \4�Ar~�   Ar}�P   Ar�ZP   \3�Ar��   Ar�ZP   Ar��P   \0GAr���   Ar��P   Ar�<P   \,�Ar�a�   Ar�<P   Ar��P   \(�Ar���   Ar��P   Ar�P   \$\Ar�C�   Ar�P   Ar��P   \%Ar���   Ar��P   Ar� P   \+�Ar�%�   Ar� P   Ar�qP   \0�Ar���   Ar�qP   Ar��P   \8�Ar��   Ar��P   Ar�SP   \?�Ar�x�   Ar�SP   Ar��P   \9�Ar���   Ar��P   Ar�5P   \4�Ar�Z�   Ar�5P   Ar��P   \*Ar���   Ar��P   Ar�P   \)�Ar�<�   Ar�P   Ar��P   \&~Ar���   Ar��P   Ar��P   \&iAr��   Ar��P   Ar�jP   \+^Ar���   Ar�jP   Ar��P   \8�Ar� �   Ar��P   Ar�LP   \EpAr�q�   Ar�LP   Ar��P   \H
Ar���   Ar��P   Ar�.P   \@�Ar�S�   Ar�.P   Ar��P   \;�Ar���   Ar��P   Ar�P   \2�Ar�5�   Ar�P   Ar��P   \*4Ar���   Ar��P   Ar��P   \'wAr��   Ar��P   Ar�cP   \&�Ar���   Ar�cP   Ar��P   \&�Ar���   Ar��P   Ar�EP   \+�Ar�j�   Ar�EP   ArĶP   \3DAr���   ArĶP   Ar�'P   \7Ar�L�   Ar�'P   ArɘP   \7�Arɽ�   ArɘP   Ar�	P   \7�Ar�.�   Ar�	P   Ar�zP   \2�ArΟ�   Ar�zP   Ar��P   \1�Ar��   Ar��P   Ar�\P   \+�ArӁ�   Ar�\P   Ar��P   \(6Ar���   Ar��P   Ar�>P   \(>Ar�c�   Ar�>P   ArگP   \'YAr���   ArگP   Ar� P   \,vAr�E�   Ar� P   ArߑP   \8�Ar߶�   ArߑP   Ar�P   \>YAr�'�   Ar�P   Ar�sP   \>�Ar��   Ar�sP   Ar��P   \?�Ar�	�   Ar��P   Ar�UP   \61Ar�z�   Ar�UP   Ar��P   \5�Ar���   Ar��P   Ar�7P   \*�Ar�\�   Ar�7P   Ar�P   \&vAr���   Ar�P   Ar�P   \&�Ar�>�   Ar�P   Ar��P   \(�Ar���   Ar��P   Ar��P   \2�Ar� �   Ar��P   Ar�lP   \?'Ar���   Ar�lP   Ar��P   \?�Ar��   Ar��P   Ar�NP   \>$Ar�s�   Ar�NP   As�P   \8�As��   As�P   As0P   \/�AsU�   As0P   As�P   \(LAs��   As�P   As	P   \'As	7�   As	P   As�P   \$As��   As�P   As�P   \'hAs�   As�P   AseP   \0�As��   AseP   As�P   \9�As��   As�P   AsGP   \>!Asl�   AsGP   As�P   \=As��   As�P   As)P   \>AsN�   As)P   As�P   \1�As��   As�P   AsP   \*�As0�   AsP   As!|P   \%}As!��   As!|P   As#�P   \#�As$�   As#�P   As&^P   \ �As&��   As&^P   As(�P   \ uAs(��   As(�P   As+@P   \ )As+e�   As+@P   As-�P   \%As-��   As-�P   As0"P   \+AAs0G�   As0"P   As2�P   \0�As2��   As2�P   As5P   \1qAs5)�   As5P   As7uP   \3�As7��   As7uP   As9�P   \.�As:�   As9�P   As<WP   \%�As<|�   As<WP   As>�P   \#iAs>��   As>�P   AsA9P   \!�AsA^�   AsA9P   AsC�P   \"AsC��   AsC�P   AsFP   \"�AsF@�   AsFP   AsH�P   \*2AsH��   AsH�P   AsJ�P   \3AsK"�   AsJ�P   AsMnP   \9ZAsM��   AsMnP   AsO�P   \:�AsP�   AsO�P   AsRPP   \6/AsRu�   AsRPP   AsT�P   \19AsT��   AsT�P   AsW2P   \+�AsWW�   AsW2P   AsY�P   \(EAsY��   AsY�P   As\P   \%�As\9�   As\P   As^�P   \#BAs^��   As^�P   As`�P   \!Asa�   As`�P   AscgP   \!XAsc��   AscgP   Ase�P   \%VAse��   Ase�P   AshIP   \.7Ashn�   AshIP   Asj�P   \/4Asj��   Asj�P   Asm+P   \4lAsmP�   Asm+P   Aso�P   \/uAso��   Aso�P   AsrP   \/JAsr2�   AsrP   Ast~P   \*�Ast��   Ast~P   Asv�P   \%�Asw�   Asv�P   Asy`P   \ �Asy��   Asy`P   As{�P   \cAs{��   As{�P   As~BP   \�As~g�   As~BP   As��P   \"3As���   As��P   As�$P   \)�As�I�   As�$P   As��P   \/�As���   As��P   As�P   \6As�+�   As�P   As�wP   \6�As���   As�wP   As��P   \1�As��   As��P   As�YP   \*�As�~�   As�YP   As��P   \'As���   As��P   As�;P   \#FAs�`�   As�;P   As��P   \ [As���   As��P   As�P   \!As�B�   As�P   As��P   \%�As���   As��P   As��P   \+]As�$�   As��P   As�pP   \+�As���   As�pP   As��P   \,�As��   As��P   As�RP   \-�As�w�   As�RP   As��P   \)�As���   As��P   As�4P   \$�As�Y�   As�4P   As��P   \!�As���   As��P   As�P   \ _As�;�   As�P   As��P   \�As���   As��P   As��P   \ �As��   As��P   As�iP   \&�As���   As�iP   As��P   \,�As���   As��P   As�KP   \07As�p�   As�KP   As��P   \/~As���   As��P   As�-P   \+�As�R�   As�-P   AsP   \&�As���   AsP   As�P   \% As�4�   As�P   AsǀP   \"�Asǥ�   AsǀP   As��P   \�As��   As��P   As�bP   \�Aṡ�   As�bP   As��P   \�As���   As��P   As�DP   \�As�i�   As�DP   AsӵP   \"�As���   AsӵP   As�&P   \'BAs�K�   As�&P   AsؗP   \*UAsؼ�   AsؗP   As�P   \* As�-�   As�P   As�yP   \*�Asݞ�   As�yP   As��P   \'�As��   As��P   As�[P   \%�As��   As�[P   As��P   \!�As���   As��P   As�=P   \�As�b�   As�=P   As�P   \,As���   As�P   As�P   \�As�D�   As�P   As�P   \!�As��   As�P   As�P   \($As�&�   As�P   As�rP   \,As��   As�rP   As��P   \2`As��   As��P   As�TP   \0�As�y�   As�TP   As��P   \,?As���   As��P   As�6P   \':As�[�   As�6P   As��P   \#PAs���   As��P   AtP   \!�At=�   AtP   At�P   \�At��   At�P   At�P   \!�At�   At�P   At	kP   \%�